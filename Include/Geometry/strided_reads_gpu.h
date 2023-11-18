/***************************************************************************\
* Copyright (c) 2023, Sofie Martins                                         *
* All rights reserved.                                                      *
\***************************************************************************/

/// Headerfile for:
/// - strided_reads.cu

#ifdef __cplusplus

#ifndef STRIDED_READS_GPU_HPP
#define STRIDED_READS_GPU_HPP

#ifdef FIXED_STRIDE
#define THREADSIZE 32
#else
#define THREADSIZE 1
#endif

#include "libhr_core.h"

enum DIRECTION { UP = 0, DOWN = 1 };

__host__ __device__ static int set_block_size(int L, int b) {
    //keep decreasing b to find a good value
    while (b > 1) {
        if (L % b == 0) { return b; }
        b--;
    }
    return 1;
}

__host__ __device__ static size_t strided_index(int ix, const int comp, const int dim, const int field_dim, const int n_components, int *blk_vol_out) {
#ifdef __CUDA_ARCH__
    int b0 = set_block_size(T_GPU, BLK_T);
    int b1 = set_block_size(X_GPU, BLK_X);
    int b2 = set_block_size(Y_GPU, BLK_Y);
    int b3 = set_block_size(Z_GPU, BLK_Z);
#else
    int b0 = set_block_size(0, BLK_T); // TODO: it cannot find T, X, Y, Z. Why?
    int b1 = set_block_size(0, BLK_X); // the host code is only necessary for testing Memory/Geometry
    int b2 = set_block_size(0, BLK_Y);
    int b3 = set_block_size(0, BLK_Z);
#endif

    int blk_vol = b0*b1*b2*b3 / 2;
    *blk_vol_out = blk_vol;

    int inner_index = ix % blk_vol;
    int block_index = ix / blk_vol;

    //printf("blk_vol=%d, dim=%d, field_dim=%d, ncomp=%d, blk_idx=%d, inner_idx=%d, offset=%d, block_idxmvmt=%d, inner_idxmvmt=%d, compmvmt=%d\n", 
//    blk_vol, dim, field_dim, n_components, block_index, inner_index, ((block_index / THREADSIZE) * THREADSIZE) * dim * field_dim * blk_vol, (block_index % THREADSIZE), 
 //   inner_index * THREADSIZE, ((comp)*n_components) * (THREADSIZE) * blk_vol);
    size_t iz = ((block_index / THREADSIZE) * THREADSIZE) * dim * field_dim * blk_vol;
    iz += (block_index % THREADSIZE);
    iz += inner_index * THREADSIZE;
    iz += ((comp)*n_components) * (THREADSIZE) * blk_vol;
    //return ((ix / THREADSIZE) * THREADSIZE) * dim * field_dim + (ix % THREADSIZE) + ((comp)*n_components) * (THREADSIZE);
    return iz;
}

template <typename REAL, typename FIELD_TYPE, typename SITE_TYPE>
__host__ __device__ void read_gpu(int stride, SITE_TYPE *s, const FIELD_TYPE *in, size_t ix, int comp, int dim) {
    const int field_dim = sizeof(FIELD_TYPE) / sizeof(REAL);
    const int n_components = sizeof(SITE_TYPE) / sizeof(REAL);
#ifdef FIXED_STRIDE
    int blk_vol;
    size_t iz = strided_index(ix, comp, dim, field_dim, n_components, &blk_vol);
    //size_t iz = ((ix / THREADSIZE) * THREADSIZE) * dim * field_dim + (ix % THREADSIZE) + ((comp)*n_components) * (THREADSIZE);
    const int _stride = THREADSIZE * blk_vol;
#else
    size_t iz = ix + ((comp)*n_components) * (THREADSIZE);
    const int _stride = stride;
#endif
    REAL *in_cpx = (REAL *)in;
    REAL *in_comp_cpx = (REAL *)s;
    for (int i = 0; i < n_components; ++i) {
        in_comp_cpx[i] = in_cpx[iz];
        iz += _stride;
    }
}

template <typename REAL, typename FIELD_TYPE, typename SITE_TYPE>
__host__ __device__ void write_gpu(int stride, SITE_TYPE *s, FIELD_TYPE *out, size_t ix, int comp, int dim) {
    const int field_dim = sizeof(FIELD_TYPE) / sizeof(REAL);
    const int n_components = sizeof(SITE_TYPE) / sizeof(REAL);
#ifdef FIXED_STRIDE
    int blk_vol;
    size_t iz = strided_index(ix, comp, dim, field_dim, n_components, &blk_vol);
   // size_t iz = ((ix / THREADSIZE) * THREADSIZE) * dim * field_dim + (ix % THREADSIZE) + (comp)*n_components * (THREADSIZE);
    const int _stride = THREADSIZE * blk_vol;
#else
    size_t iz = ix + ((comp)*n_components) * (THREADSIZE);
    const int _stride = stride;
#endif
    REAL *out_cpx = (REAL *)out;
    REAL *out_comp_cpx = (REAL *)s;
    for (int i = 0; i < n_components; ++i) {
        out_cpx[iz] = out_comp_cpx[i];
        iz += _stride;
    }
}

template <typename REAL, typename VECTOR_TYPE, typename SITE_TYPE>
__device__ void in_spinor_field(VECTOR_TYPE *v, SITE_TYPE *in, int iy, int comp) {
    read_gpu<REAL>(0, v, in, iy, comp, 1);
}

template <typename REAL, typename GAUGE_TYPE>
__device__ void in_gauge_field(GAUGE_TYPE *u, const GAUGE_TYPE *in, int ix, int iy, int comp, int dir) {
    if (dir == UP) {
        read_gpu<REAL>(0, u, in, ix, comp, 4);
    } else if (dir == DOWN) {
        read_gpu<REAL>(0, u, in, iy, comp, 4);
    }
}

template <typename REAL, typename SITE_TYPE> __device__ void write_out_spinor_field(SITE_TYPE *r, SITE_TYPE *in, int ix) {
    write_gpu<REAL>(0, r, in, ix, 0, 1);
}

template <typename REAL, typename FIELD_TYPE, typename SITE_TYPE>
__host__ __device__ void write_assign_gpu(int stride, SITE_TYPE *s, FIELD_TYPE *out, int ix, int comp, int dim) {
    const int field_dim = sizeof(FIELD_TYPE) / sizeof(REAL);
    const int n_components = sizeof(SITE_TYPE) / sizeof(REAL);
#ifdef FIXED_STRIDE
    int iz = ((ix / THREADSIZE) * THREADSIZE) * dim * field_dim + (ix % THREADSIZE) + (comp)*n_components * (THREADSIZE);
    const int _stride = THREADSIZE;
#else
    int iz = ix + ((comp)*n_components) * (THREADSIZE);
    const int _stride = stride;
#endif
    REAL *out_cpx = (REAL *)out;
    REAL *out_comp_cpx = (REAL *)s;
    for (int i = 0; i < n_components; ++i) {
        out_cpx[iz] += out_comp_cpx[i];
        iz += _stride;
    }
}

#endif
#endif

#ifndef STRIDED_READS_GPU_H
#define STRIDED_READS_GPU_H

#ifdef __cplusplus
extern "C" {
#endif

#define _FIELD_TYPE spinor_field
#define _SITE_TYPE suNf_spinor
#define _FIELD_DIM 1
#define _REAL double
#include "TMPL/strided_reads_gpu.h.tmpl"

#define _FIELD_TYPE spinor_field_flt
#define _SITE_TYPE suNf_spinor_flt
#define _FIELD_DIM 1
#define _REAL float
#include "TMPL/strided_reads_gpu.h.tmpl"

#define _FIELD_TYPE scalar_field
#define _SITE_TYPE double
#define _FIELD_DIM 1
#define _REAL double
#include "TMPL/strided_reads_gpu.h.tmpl"

#define _FIELD_TYPE suNg_field
#define _SITE_TYPE suNg
#define _FIELD_DIM 4
#define _REAL double
#include "TMPL/strided_reads_gpu.h.tmpl"

#define _FIELD_TYPE suNg_field_flt
#define _SITE_TYPE suNg_flt
#define _FIELD_DIM 4
#define _REAL float
#include "TMPL/strided_reads_gpu.h.tmpl"

#define _FIELD_TYPE suNf_field
#define _SITE_TYPE suNf
#define _FIELD_DIM 4
#define _REAL double
#include "TMPL/strided_reads_gpu.h.tmpl"

#define _FIELD_TYPE suNf_field_flt
#define _SITE_TYPE suNf_flt
#define _FIELD_DIM 4
#define _REAL float
#include "TMPL/strided_reads_gpu.h.tmpl"

#define _FIELD_TYPE suNg_scalar_field
#define _SITE_TYPE suNg_vector
#define _FIELD_DIM 1
#define _REAL double
#include "TMPL/strided_reads_gpu.h.tmpl"

#define _FIELD_TYPE suNg_av_field
#define _SITE_TYPE suNg_algebra_vector
#define _FIELD_DIM 4
#define _REAL double
#include "TMPL/strided_reads_gpu.h.tmpl"

#define _FIELD_TYPE gtransf
#define _SITE_TYPE suNg
#define _FIELD_DIM 1
#define _REAL double
#include "TMPL/strided_reads_gpu.h.tmpl"

#define _FIELD_TYPE ldl_field
#define _SITE_TYPE ldl_t
#define _FIELD_DIM 1
#define _REAL double
#include "TMPL/strided_reads_gpu.h.tmpl"

#define _FIELD_TYPE clover_term
#define _SITE_TYPE suNfc
#define _FIELD_DIM 4
#define _REAL double
#include "TMPL/strided_reads_gpu.h.tmpl"

#define _FIELD_TYPE clover_force
#define _SITE_TYPE suNf
#define _FIELD_DIM 6
#define _REAL double
#include "TMPL/strided_reads_gpu.h.tmpl"

#define _FIELD_TYPE staple_field
#define _SITE_TYPE suNg
#define _FIELD_DIM 3
#define _REAL double
#include "TMPL/strided_reads_gpu.h.tmpl"

#ifdef __cplusplus
}
#endif

#endif