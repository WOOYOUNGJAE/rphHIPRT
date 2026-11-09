//////////////////////////////////////////////////////////////////////////////////////////
//
//  Copyright (c) 2024 Advanced Micro Devices, Inc. All Rights Reserved.
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.
//
//////////////////////////////////////////////////////////////////////////////////////////

#pragma once
#include <hiprt/hiprt_common.h>
#include <hiprt/hiprt_math.h>
#include <hiprt/impl/BvhNode.h>
#include <hiprt/impl/Obb.h>
#include <hiprt/impl/Triangle.h>

// Software implementation of the RTIP 3.1 image_bvh8_intersect_ray instruction, used when HIPRT is built with
// FORCE_RTIP31_EMULATION. It follows the instruction emulation of AMD GPURT (src/shaders/rtip3_opcode_emulation.hlsl) to
// behave like the hardware: same box growth, same child ordering and same triangle test.
namespace hiprt
{
namespace rtip31
{
// Must match boxGrowUlp in TraversalBase::packDescriptor().
static constexpr uint32_t BoxGrowUlp     = 6;
static constexpr float    BoxGrowEpsilon = 5.960464478e-8f; // 2^-24
static constexpr float    BoxGrowFactor  = 1.0f + BoxGrowUlp * BoxGrowEpsilon;

// The box sort heuristic the hardware reads from the descriptor; the values are those of hiprtTraversalHint.
static constexpr uint32_t BoxSortLargest  = 1;
static constexpr uint32_t BoxSortMidPoint = 2;
static constexpr uint32_t BoxSortDisabled = 3;

static constexpr uint32_t PairResultSize       = 9;
static constexpr uint32_t NavigationEndOfPair  = 0;
static constexpr uint32_t NavigationEndOfNode  = 1;
static constexpr uint32_t NavigationEndOfRange = 3;

struct BoxInterval
{
	float m_minT;
	float m_maxT;
	float m_minOfIntervals;
	float m_maxOfIntervals;
};

struct DecodedChildren
{
	uint32_t m_indices[BoxNode::BranchingFactor];
	float3   m_min[BoxNode::BranchingFactor];
	float3   m_max[BoxNode::BranchingFactor];
};

// A bit test rather than a comparison, so that it holds under any floating point optimization mode.
[[nodiscard]] HIPRT_DEVICE HIPRT_INLINE bool isNan( const float value )
{
	return ( as_uint( value ) & 0x7fffffffu ) > 0x7f800000u;
}

[[nodiscard]] HIPRT_DEVICE HIPRT_INLINE float infinity() { return as_float( 0x7f800000u ); }

[[nodiscard]] HIPRT_DEVICE HIPRT_INLINE float dequantize( const float origin, const uint32_t exponent, const uint32_t plane )
{
	constexpr int32_t QuantizationBits = 12;

	uint32_t result = 0u;
	if ( plane != 0u )
	{
		const int32_t position       = 31 - static_cast<int32_t>( clz( plane ) ) - QuantizationBits;
		const int32_t signedExponent = position + static_cast<int32_t>( exponent );
		if ( signedExponent > 0 )
		{
			result = ( plane << ( 23 - QuantizationBits - position ) ) & 0x7fffffu;
			result |= ( static_cast<uint32_t>( signedExponent ) << 23 ) & 0x7fffffffu;
		}
	}

	return as_float( result ) + origin;
}

template <typename Node>
[[nodiscard]] HIPRT_DEVICE HIPRT_INLINE DecodedChildren
decodeChildren( const Node& node, uint32_t internalOffset, uint32_t primitiveOffset, const bool instanceMaskCulling )
{
	const float    nan        = as_float( 0x7fc00000u );
	const uint32_t validCount = node.getChildCount();

	DecodedChildren children;
	for ( uint32_t i = 0; i < BoxNode::BranchingFactor; ++i )
	{
		if ( i >= validCount )
		{
			children.m_indices[i] = InvalidValue;
			children.m_min[i]     = make_float3( nan );
			children.m_max[i]     = make_float3( nan );
			continue;
		}

		const ChildInfo& info = node.m_childInfos[i];

		if ( info.m_nodeType == BoxType )
		{
			children.m_indices[i] = internalOffset;
			internalOffset += info.m_nodeRange << 4;
		}
		else
		{
			children.m_indices[i] = primitiveOffset | info.m_nodeType;
			primitiveOffset += info.m_nodeRange << 4;
		}

		// The ray mask of the traversal is always 0xff.
		if ( instanceMaskCulling && info.m_instanceMask == 0 ) children.m_indices[i] = InvalidValue;

		children.m_min[i].x = dequantize( node.m_origin.x, node.m_exponentX, info.m_minX );
		children.m_min[i].y = dequantize( node.m_origin.y, node.m_exponentY, info.m_minY );
		children.m_min[i].z = dequantize( node.m_origin.z, node.m_exponentZ, info.m_minZ );
		children.m_max[i].x = dequantize( node.m_origin.x, node.m_exponentX, info.m_maxX + 1 );
		children.m_max[i].y = dequantize( node.m_origin.y, node.m_exponentY, info.m_maxY + 1 );
		children.m_max[i].z = dequantize( node.m_origin.z, node.m_exponentZ, info.m_maxZ + 1 );
	}

	return children;
}

[[nodiscard]] HIPRT_DEVICE HIPRT_INLINE BoxInterval
intersectBox( const float3& origin, const float3& invD, const float3& boxMin, const float3& boxMax, const float maxT )
{
	const float3 planeMin = ( boxMin - origin ) * invD;
	const float3 planeMax = ( boxMax - origin ) * invD;

	float3 minInterval;
	float3 maxInterval;
	minInterval.x = invD.x >= 0.0f ? planeMin.x : planeMax.x;
	maxInterval.x = invD.x >= 0.0f ? planeMax.x : planeMin.x;
	minInterval.y = invD.y >= 0.0f ? planeMin.y : planeMax.y;
	maxInterval.y = invD.y >= 0.0f ? planeMax.y : planeMin.y;
	minInterval.z = invD.z >= 0.0f ? planeMin.z : planeMax.z;
	maxInterval.z = invD.z >= 0.0f ? planeMax.z : planeMin.z;

	BoxInterval interval;
	interval.m_minOfIntervals = fmaxf( fmaxf( minInterval.x, minInterval.y ), minInterval.z );
	interval.m_maxOfIntervals = fminf( fminf( maxInterval.x, maxInterval.y ), maxInterval.z );
	interval.m_minT           = fmaxf( interval.m_minOfIntervals, 0.0f );
	interval.m_maxT           = fminf( interval.m_maxOfIntervals, maxT );

	if ( isNan( interval.m_minOfIntervals ) || isNan( interval.m_maxOfIntervals ) )
	{
		interval.m_minT = infinity();
		interval.m_maxT = -infinity();
	}

	if ( isNan( interval.m_minOfIntervals ) ) interval.m_minOfIntervals = 0.0f;
	if ( isNan( interval.m_maxOfIntervals ) ) interval.m_maxOfIntervals = infinity();

	return interval;
}

HIPRT_DEVICE HIPRT_INLINE void sortChildPair(
	const uint32_t a,
	const uint32_t b,
	uint32_t ( &extendedKeys )[BoxNode::BranchingFactor],
	uint32_t ( &children )[BoxNode::BranchingFactor],
	float ( &keys )[BoxNode::BranchingFactor] )
{
	const bool swapPair = extendedKeys[a] != extendedKeys[b] ? extendedKeys[b] < extendedKeys[a] : keys[b] < keys[a];
	if ( swapPair )
	{
		const uint32_t extendedKey = extendedKeys[a];
		const uint32_t child       = children[a];
		const float    key         = keys[a];

		extendedKeys[a] = extendedKeys[b];
		children[a]     = children[b];
		keys[a]         = keys[b];

		extendedKeys[b] = extendedKey;
		children[b]     = child;
		keys[b]         = key;
	}
}

HIPRT_DEVICE HIPRT_INLINE void intersectChildren(
	const DecodedChildren& children,
	const float3&          origin,
	const float3&          invD,
	const float            maxT,
	const uint32_t         boxSortHeuristic,
	uint32_t ( &result )[BoxNode::BranchingFactor] )
{
	constexpr uint32_t MaxLeafNodeType = 3;

	float    keys[BoxNode::BranchingFactor];
	uint32_t extendedKeys[BoxNode::BranchingFactor];

	HIPRT_UNROLL
	for ( uint32_t i = 0; i < BoxNode::BranchingFactor; ++i )
	{
		const BoxInterval interval = intersectBox( origin, invD, children.m_min[i], children.m_max[i], maxT );

		keys[i] = interval.m_minT;
		if ( boxSortHeuristic == BoxSortLargest )
			keys[i] = interval.m_minT - interval.m_maxT;
		else if ( boxSortHeuristic == BoxSortMidPoint )
			keys[i] = interval.m_minOfIntervals + interval.m_maxOfIntervals;

		// Equality is a hit and the interval is grown, so that rounding errors cannot make the ray miss a box.
		result[i] = interval.m_minT <= interval.m_maxT * BoxGrowFactor ? children.m_indices[i] : InvalidValue;
	}

	// Children sharing a node are in contiguous slots, only the first one is traversed.
	uint32_t previous = result[0];
	for ( uint32_t i = 1; i < BoxNode::BranchingFactor; ++i )
	{
		if ( result[i] == previous )
			result[i] = InvalidValue;
		else
			previous = result[i];
	}

	HIPRT_UNROLL
	for ( uint32_t i = 0; i < BoxNode::BranchingFactor; ++i )
	{
		const uint32_t nodeType = result[i] & 7;

		extendedKeys[i] = result[i] == InvalidValue ? 4 : 0;
		extendedKeys[i] |= nodeType > MaxLeafNodeType ? 2 : 0;
		extendedKeys[i] |= nodeType != 1 ? 1 : 0;
	}

	if ( boxSortHeuristic != BoxSortDisabled )
	{
		sortChildPair( 0, 2, extendedKeys, result, keys );
		sortChildPair( 1, 3, extendedKeys, result, keys );
		sortChildPair( 4, 6, extendedKeys, result, keys );
		sortChildPair( 5, 7, extendedKeys, result, keys );

		sortChildPair( 0, 1, extendedKeys, result, keys );
		sortChildPair( 2, 3, extendedKeys, result, keys );
		sortChildPair( 4, 5, extendedKeys, result, keys );
		sortChildPair( 6, 7, extendedKeys, result, keys );

		sortChildPair( 1, 2, extendedKeys, result, keys );
		sortChildPair( 5, 6, extendedKeys, result, keys );

		sortChildPair( 0, 4, extendedKeys, result, keys );
		sortChildPair( 1, 5, extendedKeys, result, keys );
		sortChildPair( 2, 6, extendedKeys, result, keys );
		sortChildPair( 3, 7, extendedKeys, result, keys );

		sortChildPair( 2, 4, extendedKeys, result, keys );
		sortChildPair( 3, 5, extendedKeys, result, keys );

		sortChildPair( 1, 2, extendedKeys, result, keys );
		sortChildPair( 3, 4, extendedKeys, result, keys );
		sortChildPair( 5, 6, extendedKeys, result, keys );
	}
}

// The evaluation order is the one of the hardware, which matters for the rounding.
[[nodiscard]] HIPRT_DEVICE HIPRT_INLINE float
transformComponent( const float m0, const float m1, const float m2, const float m3, const float3& value )
{
	return value.x * m0 + ( value.y * m1 + ( value.z * m2 + m3 ) );
}

HIPRT_DEVICE HIPRT_INLINE void intersectBoxNode(
	const BoxNode&  node,
	const hiprtRay& ray,
	const float3&   invD,
	const uint32_t  boxSortHeuristic,
	uint32_t ( &result )[BoxNode::BranchingFactor] )
{
	float3 localOrigin = ray.origin;
	float3 localInvD   = invD;

	// The boxes of an oriented node are expressed in the rotated frame.
	if ( node.m_matrixId != NoRotationIndex )
	{
		float R[3][3];
		setRotationMatrix( MatrixIdToIndex[node.m_matrixId], R );

		float3 localDirection;
		localOrigin.x    = transformComponent( R[0][0], R[0][1], R[0][2], 0.0f, ray.origin );
		localOrigin.y    = transformComponent( R[1][0], R[1][1], R[1][2], 0.0f, ray.origin );
		localOrigin.z    = transformComponent( R[2][0], R[2][1], R[2][2], 0.0f, ray.origin );
		localDirection.x = transformComponent( R[0][0], R[0][1], R[0][2], 0.0f, ray.direction );
		localDirection.y = transformComponent( R[1][0], R[1][1], R[1][2], 0.0f, ray.direction );
		localDirection.z = transformComponent( R[2][0], R[2][1], R[2][2], 0.0f, ray.direction );
		localInvD        = 1.0f / localDirection;
	}

	const DecodedChildren children = decodeChildren( node, node.m_boxNodeBase | BoxType, node.m_primNodeBase, true );
	intersectChildren( children, localOrigin, localInvD, ray.maxT, boxSortHeuristic, result );
}

HIPRT_DEVICE HIPRT_INLINE void
intersectTriangle( const hiprtRay& ray, const Triangle& triangle, const uint32_t primIndex, uint32_t* result )
{
	const float3 e1 = triangle.m_vertices[1] - triangle.m_vertices[0];
	const float3 e2 = triangle.m_vertices[2] - triangle.m_vertices[0];
	const float3 e3 = ray.origin - triangle.m_vertices[0];
	const float3 s1 = cross( ray.direction, e2 );
	const float3 s2 = cross( e3, e1 );

	const float denominator = dot( s1, e1 );
	const float t           = dot( e2, s2 ) / denominator;
	const float u           = dot( e3, s1 ) / denominator;
	const float v           = dot( ray.direction, s2 ) / denominator;

	const bool missed = u < 0.0f || u > 1.0f || v < 0.0f || u + v > 1.0f || t < 0.0f;

	result[0] = as_uint( missed ? infinity() : t );
	result[1] = as_uint( u );
	result[2] = as_uint( v );
	result[3] = ( primIndex << 1u ) | ( as_uint( denominator ) >> 31u );
}

// Fills the registers of the instruction: t, u, v and primitive index of both triangles, then the navigation bits.
HIPRT_DEVICE HIPRT_INLINE void intersectTrianglePair(
	const hiprtRay&           ray,
	const TrianglePacketNode& node,
	const uint32_t            pairIndex,
	uint32_t ( &result )[PairResultSize] )
{
	intersectTriangle( ray, node.fetchTriangle( pairIndex, 0 ), node.getPrimIndex( pairIndex, 0 ), &result[0] );

	const uint4 secondTriangle = node.readDescriptor( pairIndex, 1 );
	if ( secondTriangle.x != 0 || secondTriangle.y != 0 || secondTriangle.z != 0 )
	{
		intersectTriangle( ray, node.fetchTriangle( pairIndex, 1 ), node.getPrimIndex( pairIndex, 1 ), &result[4] );
	}
	else
	{
		result[4] = as_uint( infinity() );
		result[5] = as_uint( 1.0f );
		result[6] = as_uint( 1.0f );
		result[7] = 0u;
	}

	if ( node.isRangeEnd( pairIndex ) )
		result[8] = NavigationEndOfRange;
	else if ( pairIndex + 1 == node.getTrianglePairCount() )
		result[8] = NavigationEndOfNode;
	else
		result[8] = NavigationEndOfPair;
}

// Transforms the ray to the object space of the instance. Returns false if the instance is culled, which happens when
// the transformed ray misses all the root boxes stored in the instance node.
[[nodiscard]] HIPRT_DEVICE HIPRT_INLINE bool intersectInstanceNode(
	const InstanceNode& node, const hiprtRay& ray, const uint32_t boxSortHeuristic, float3& origin, float3& direction )
{
	// The ray mask of the traversal is always 0xff.
	if ( node.m_hwMask == 0 ) return false;

	const float ( &m )[3][4] = node.m_matrix;

	origin.x    = transformComponent( m[0][0], m[0][1], m[0][2], m[0][3], ray.origin );
	origin.y    = transformComponent( m[1][0], m[1][1], m[1][2], m[1][3], ray.origin );
	origin.z    = transformComponent( m[2][0], m[2][1], m[2][2], m[2][3], ray.origin );
	direction.x = transformComponent( m[0][0], m[0][1], m[0][2], 0.0f, ray.direction );
	direction.y = transformComponent( m[1][0], m[1][1], m[1][2], 0.0f, ray.direction );
	direction.z = transformComponent( m[2][0], m[2][1], m[2][2], 0.0f, ray.direction );

	uint32_t              result[BoxNode::BranchingFactor];
	const DecodedChildren children = decodeChildren( node, BoxType, 0u, false );
	intersectChildren( children, origin, 1.0f / direction, ray.maxT, boxSortHeuristic, result );

	for ( uint32_t i = 0; i < InstanceNode::BranchingFactor; ++i )
		if ( result[i] != InvalidValue ) return true;

	return false;
}
} // namespace rtip31
} // namespace hiprt
