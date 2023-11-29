//
// Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//

#include <optix.h>

#include "raycasting_mesh.h"
#include "helpers.h"
#include "vec_math.h"


extern "C" {
__constant__ Params params;
}


extern "C" __global__ void __raygen__from_buffer()
{
    const uint3        idx        = optixGetLaunchIndex();
    const uint3        dim        = optixGetLaunchDimensions();
    const unsigned int linear_idx = idx.x;

    unsigned int h, x, y, z;

    float3 origin = params.origins[linear_idx];
    float3 dir = params.dirs[linear_idx];
    float tmin = params.tmins[linear_idx];
    float tmax = params.tmaxs[linear_idx];

    optixTrace( params.trav_handle, origin, dir, tmin, tmax, 0.0f, OptixVisibilityMask( 1 ),
                OPTIX_RAY_FLAG_NONE, RAY_TYPE_RADIANCE, RAY_TYPE_COUNT, RAY_TYPE_RADIANCE, h, x, y, z );

    
    params.hits[linear_idx].x = __uint_as_float( x );
    params.hits[linear_idx].y = __uint_as_float( y );
    params.hits[linear_idx].z = __uint_as_float( z );
    params.hits[linear_idx].w = __uint_as_float( h );
}

extern "C" __global__ void __miss__buffer_miss()
{
    optixSetPayload_0( __float_as_uint( 0.0f ) );
    optixSetPayload_1( __float_as_uint( 0.0f ) );
    optixSetPayload_2( __float_as_uint( 0.0f ) );
    optixSetPayload_3( __float_as_uint( 0.0f ) );
}


extern "C" __global__ void __closesthit__buffer_hit()
{
    const float3 pos = optixGetWorldRayOrigin() + optixGetWorldRayDirection() * optixGetRayTmax();
    optixSetPayload_0( __float_as_uint( 1.0f ) );
    optixSetPayload_1( __float_as_uint( pos.x) );
    optixSetPayload_2( __float_as_uint( pos.y ) );
    optixSetPayload_3( __float_as_uint( pos.z ) );
}

extern "C" __global__ void __anyhit__buffer_hit()
{
    optixSetPayload_0( __float_as_uint( 1.0f ) );
}

extern "C" __global__ void __miss__buffer_miss_any()
{
    optixSetPayload_0( __float_as_uint( 0.0f ) );
}
