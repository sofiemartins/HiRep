/***************************************************************************\
* Copyright (c) 2023, Sofie Martins                                         *
* All rights reserved.                                                      *
\***************************************************************************/

/*******************************************************************************
*
* File plaquette.c
*
* Routines for the average plaquette
*
*******************************************************************************/

// With patterns from linear_algebra_gpu.cu + templates/kernels
// based on avr_plaquette.c

#include "libhr_core.h"
#include "update.h"
#include "inverters.h"
#include "./gauge_gpu_kernels.hpp"

#ifdef WITH_GPU

static int *timeslices;
static int init = 0;

#ifdef PLAQ_WEIGHTS
#define PLAQ_WEIGHT_ARG , plaq_weight
#define PLAQ_WEIGHT_ARG_DEF , double *plaq_weight
#else
#define PLAQ_WEIGHT_ARG
#define PLAQ_WEIGHT_ARG_DEF
#endif

template <class T> T global_sum_gpu(T *vector, int size);

#define _CUDA_FOR(s, ixp, body)                                                        \
    do {                                                                               \
        _PIECE_FOR((s)->type, (ixp)) {                                                 \
            int N = (s)->type->master_end[(ixp)] - (s)->type->master_start[(ixp)] + 1; \
            unsigned int grid_size = (N - 1) / BLOCK_SIZE_LINEAR_ALGEBRA + 1;          \
            body;                                                                      \
            CudaCheckError();                                                          \
        }                                                                              \
    } while (0)

double avr_plaquette_gpu(suNg_field *gauge) {
    double res = 0.0;
    double *resPiece;

    complete_sendrecv_suNg_field(gauge);

    _CUDA_FOR(u_gauge, ixp, resPiece = alloc_double_sum_field(N);
              (_avr_plaquette<<<grid_size, BLOCK_SIZE_LINEAR_ALGEBRA, 0, 0>>>(gauge->gpu_ptr, resPiece, iup_gpu, N,
                                                                              gauge->type->master_start[ixp] PLAQ_WEIGHT_ARG));
              res += global_sum_gpu(resPiece, N););

#ifdef WITH_MPI
    global_sum(&res, 1);
#endif

#ifdef BC_T_OPEN
    res /= 6.0 * NG * GLB_VOL3 * (GLB_T - 1);
#elif BC_T_SF
    res /= 6.0 * NG * GLB_VOL3 * (GLB_T - 2);
#else
    res /= 6.0 * NG * GLB_VOLUME;
#endif

    return res;
}

void local_plaquette_gpu(suNg_field *gauge, scalar_field *s) {
    complete_sendrecv_suNg_field(gauge);

    _CUDA_FOR(gauge, ixp, int block_start = gauge->type->master_start[ixp];
              (_avr_plaquette<<<grid_size, BLOCK_SIZE_LINEAR_ALGEBRA, 0, 0>>>(gauge->gpu_ptr, s->gpu_ptr + block_start, iup_gpu,
                                                                              N, block_start PLAQ_WEIGHT_ARG)););
}

void init_avr_plaquette_time(u_gauge) {
    int nts[GLB_VOLUME];

    for (int nt = 0; nt < T; nt++) {
        for (int nx = 0; nx < X; nx++) {
            for (int ny = 0; ny < Y; ny++) {
                for (int nz = 0; nz < Z; nz++) {
                    int ix = ipt(nt, nx, ny, nz);
                    nts[ix] = nt;
                }
            }
        }
    }

    cudaMalloc((void **)&timeslices, GLB_VOLUME * sizeof(int));
    cudaMemcpy(timeslices, nts, GLB_VOLUME * sizeof(int), cudaMemcpyHostToDevice);
}

void avr_plaquette_time_gpu(suNg_field *gauge, double *plaqt, double *plaqs) {
    double *resPiece;

    for (int nt = 0; nt < GLB_T; nt++) {
        plaqt[nt] = plaqs[nt] = 0.0;
    }

    if (!init) {
        init_avr_plaquette_time(u_gauge);
        init = 1;
    }

    start_sendrecv_suNg_field(gauge);
    complete_sendrecv_suNg_field(gauge);

    _CUDA_FOR(
        u_gauge, ixp, resPiece = alloc_double_sum_field(N * GLB_T * 2); cudaMemset(resPiece, 0, N * GLB_T * 2 * sizeof(double));
        (_avr_plaquette_time<<<grid_size, BLOCK_SIZE_LINEAR_ALGEBRA, 0, 0>>>(gauge->gpu_ptr, resPiece, zerocoord[0], GLB_T,
                                                                             iup_gpu, timeslices, N, T,
                                                                             gauge->type->master_start[ixp] PLAQ_WEIGHT_ARG));
        for (int nt = 0; nt < GLB_T; nt++) {
            int tc = (zerocoord[0] + nt + GLB_T) % GLB_T;
            plaqt[tc] += global_sum_gpu(resPiece + N * tc, N) / 3.0 / NG / GLB_VOL3;
        } for (int nt = 0; nt < GLB_T; nt++) {
            int tc = (zerocoord[0] + nt + GLB_T) % GLB_T;
            plaqs[tc] += global_sum_gpu(resPiece + N * tc + N * GLB_T, N) / 3.0 / NG / GLB_VOL3;
        });

    for (int nt = 0; nt < GLB_T; nt++) {
        global_sum(&plaqt[nt], 1);
        global_sum(&plaqs[nt], 1);
    }
}

void full_plaquette_gpu(suNg_field *gauge) {
    start_sendrecv_suNg_field(gauge);
    complete_sendrecv_suNg_field(gauge);

    hr_complex pa[6];

    hr_complex r0 = 0.0;
    hr_complex r1 = 0.0;
    hr_complex r2 = 0.0;
    hr_complex r3 = 0.0;
    hr_complex r4 = 0.0;
    hr_complex r5 = 0.0;

    hr_complex *resPiece;

    _CUDA_FOR(gauge, ixp, resPiece = alloc_complex_sum_field(N * 6); cudaMemset(resPiece, 0, N * 6 * sizeof(hr_complex));
              (_full_plaquette<<<grid_size, BLOCK_SIZE_LINEAR_ALGEBRA, 0, 0>>>(gauge->gpu_ptr, resPiece, iup_gpu, N,
                                                                               gauge->type->master_start[ixp] PLAQ_WEIGHT_ARG));
              r0 += global_sum_gpu(resPiece, N); r1 += global_sum_gpu(resPiece + N, N);
              r2 += global_sum_gpu(resPiece + 2 * N, N); r3 += global_sum_gpu(resPiece + 3 * N, N);
              r4 += global_sum_gpu(resPiece + 4 * N, N); r5 += global_sum_gpu(resPiece + 5 * N, N););

    pa[0] = r0;
    pa[1] = r1;
    pa[2] = r2;
    pa[3] = r3;
    pa[4] = r4;
    pa[5] = r5;

    global_sum((double *)pa, 12);
    double *pad = (double *)pa;
    for (int k = 0; k < 12; k++) {
#ifdef BC_T_OPEN
        pad[k] /= NG * GLB_VOLUME * (GLB_T - 1) / GLB_T;
#else
        pad[k] /= NG * GLB_VOLUME;
#endif
    }

    lprintf("PLAQ", 0, "Plaq(%d,%d) = ( %f , %f )\n", 1, 0, creal(pa[0]), cimag(pa[0]));
    lprintf("PLAQ", 0, "Plaq(%d,%d) = ( %f , %f )\n", 2, 0, creal(pa[1]), cimag(pa[1]));
    lprintf("PLAQ", 0, "Plaq(%d,%d) = ( %f , %f )\n", 2, 1, creal(pa[2]), cimag(pa[2]));
    lprintf("PLAQ", 0, "Plaq(%d,%d) = ( %f , %f )\n", 3, 0, creal(pa[3]), cimag(pa[3]));
    lprintf("PLAQ", 0, "Plaq(%d,%d) = ( %f , %f )\n", 3, 1, creal(pa[4]), cimag(pa[4]));
    lprintf("PLAQ", 0, "Plaq(%d,%d) = ( %f , %f )\n", 3, 2, creal(pa[5]), cimag(pa[5]));
}
#endif

#undef PLAQ_WEIGHT_ARG
#undef PLAQ_WEIGHT_ARG_DEF

double E_gpu(suNg_field *V) {
    double res = 0.0;
    double *resPiece;
    start_sendrecv_suNg_field(V);
    complete_sendrecv_suNg_field(V);
    _CUDA_FOR(V, ixp, resPiece = alloc_double_sum_field(N);
              (_E_gpu<<<grid_size, BLOCK_SIZE, 0, 0>>>(V->gpu_ptr, resPiece, iup_gpu, N, block_start));
              res += global_sum_gpu(resPiece, N););
    res *= 2. / ((double)GLB_VOLUME);
#ifdef WITH_MPI
    global_sum(&res, 1);
#endif
    return res;
}

void E_T_gpu(double *E, suNg_field *V) {
    double *resPiece;
    memset(E, 0, GLB_T * sizeof(double));

    if (!init) {
        init_avr_plaquette_time(u_gauge);
        init = 1;
    }

    start_sendrecv_suNg_field(V);
    complete_sendrecv_suNg_field(V);

    _CUDA_FOR(
        u_gauge, ixp, resPiece = alloc_double_sum_field(N * GLB_T * 4); cudaMemset(resPiece, 0, N * GLB_T * 4 * sizeof(double));
        (_E_T_gpu<<<grid_size, BLOCK_SIZE, 0, 0>>>(V->gpu_ptr, resPiece, zerocoord[0], GLB_T, iup_gpu, timeslices, N,
                                                   block_start));
        for (int nt = 0; nt < GLB_T; nt++) {
            int gt = (zerocoord[0] + nt + GLB_T) % GLB_T;
            E[2 * gt] += global_sum_gpu(resPiece + N * gt);
            E[2 * gt + 1] += global_sum_gpu(resPiece + N * gt * 2);
        });

    E[2 * gt] /= 0.5 * (GLB_VOL3);
    E[2 * gt + 1] /= 0.5 * (GLB_VOL3);

#ifdef WITH_MPI
    global_sum(E, 2 * GLB_T);
#endif
}

double Esym_gpu(suNg_field *V) {
    double res = 0.0;
    double *resPiece;
    start_sendrecv_suNg_field(V);
    complete_sendrecv_suNg_field(V);
    _CUDA_FOR(V, ixp, resPiece = alloc_double_sum_field(N);
              (_Esym_gpu<<<grid_size, BLOCK_SIZE, 0, 0>>>(V->gpu_ptr, resPiece, iup_gpu, idn_gpu, N, block_start));
              res += global_sum_gpu(resPiece, N););
    res *= _FUND_NORM2 / ((double)GLB_VOLUME);
    global_sum(&res, 1);
    return res;
}

void Esym_T_gpu(double *E, suNg_field *V) {
    double *resPiece;
    memset(E, 0, GLB_T * sizeof(double));

    if (!init) {
        init_avr_plaquette_time(u_gauge);
        init = 1;
    }

    start_sendrecv_suNg_field(V);
    complete_sendrecv_suNg_field(V);

    _CUDA_FOR(
        u_gauge, ixp, resPiece = alloc_double_sum_field(N * GLB_T * 4); cudaMemset(resPiece, 0, N * GLB_T * 4 * sizeof(double));
        (_Esym_T_gpu<<<grid_size, BLOCK_SIZE, 0, 0>>>(V->gpu_ptr, resPiece, zerocoord[0], GLB_T, iup_gpu, idn_gpu, timeslices,
                                                      N, block_start));
        for (int nt = 0; nt < GLB_T; nt++) {
            int gt = (zerocoord[0] + nt + GLB_T) % GLB_T;
            E[2 * gt] += global_sum_gpu(resPiece + N * gt);
            E[2 * gt + 1] += global_sum_gpu(resPiece + N * gt * 2);
        });

    E[2 * gt] /= 0.5 * (GLB_VOL3);
    E[2 * gt + 1] /= 0.5 * (GLB_VOL3);

#ifdef WITH_MPI
    global_sum(E, 2 * GLB_T);
#endif
}

double topo_gpu(suNg_field *V) {
    double res = 0.0;
    double *resPiece;
    start_sendrecv_suNg_field(V);
    complete_sendrecv_suNg_field(V);
    _CUDA_FOR(V, ixp, resPiece = alloc_double_sum_field(N);
              (_topo_gpu<<<grid_size, BLOCK_SIZE, 0, 0>>>(V->gpu_ptr, resPiece, iup_gpu, idn_gpu, N, block_start));
              res += global_sum_gpu(resPiece, N););
    res *= _FUND_NORM2 / (4. * M_PI * M_PI);
    global_sum(&res, 1);
    return res;
}

double (*E)(suNg_field *V) = E_gpu;
void (*E_T)(double *E, suNg_field *V) = E_T_gpu;
double (*Esym)(suNg_field *V) = Esym_gpu;
void (*Esym_T)(double *E, suNg_field *V) = Esym_T_gpu;
double (*topo)(suNg_field *V) = topo_gpu;
double (*avr_plaquette)(suNg_field *gauge) = avr_plaquette_gpu;
void (*full_plaquette)(suNg_field *gauge) = full_plaquette_gpu;
void (*local_plaquette)(suNg_field *gauge, scalar_field *s) = local_plaquette_gpu;
void (*avr_plaquette_time)(suNg_field *gauge, double *plaqt, double *plaqs) = avr_plaquette_time_gpu;