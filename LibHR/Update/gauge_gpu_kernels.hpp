#include "geometry.h"
#include "libhr_core.h"

#ifdef PLAQ_WEIGHTS
#define PLAQ_WEIGHT_ARG , plaq_weight
#define PLAQ_WEIGHT_ARG_DEF , double *plaq_weight
#else
#define PLAQ_WEIGHT_ARG
#define PLAQ_WEIGHT_ARG_DEF
#endif

// TODO: all device functions into the header + forceinline

// Based on the implementation in avr_plaquette.c
// specify gauge as last argument if you want to
// evaluate this not on u_gauge
__device__ static double plaq_dev(int ix, int mu, int nu, suNg *gauge, int *iup_gpu PLAQ_WEIGHT_ARG_DEF) {
    int iy, iz;
    double p;
    suNg v1, v2, v3, v4, w1, w2, w3;

    iy = iup_gpu[4 * ix + mu];
    iz = iup_gpu[4 * ix + nu];

    read_gpu<double>(0, &v1, gauge, ix, mu, 4);
    read_gpu<double>(0, &v2, gauge, iy, nu, 4);
    read_gpu<double>(0, &v3, gauge, iz, mu, 4);
    read_gpu<double>(0, &v4, gauge, ix, nu, 4);

    _suNg_times_suNg(w1, v1, v2);
    _suNg_times_suNg(w2, v4, v3);
    _suNg_times_suNg_dagger(w3, w1, w2);

    _suNg_trace_re(p, w3);

#ifdef PLAQ_WEIGHTS
    if (plaq_weight == NULL) { return p; }
    return plaq_weight[ix * 16 + mu * 4 + nu] * p;
#else
    return p;
#endif
}

__device__ static void cplaq_dev(hr_complex *res, int ix, int mu, int nu, suNg *gauge, int *iup_gpu PLAQ_WEIGHT_ARG_DEF) {
    int iy, iz;
    suNg v1, v2, v3, v4, w1, w2, w3;
    double tmpre = 0.;
    double tmpim = 0.;

    iy = iup_gpu[4 * ix + mu];
    iz = iup_gpu[4 * ix + nu];

    read_gpu<double>(0, &v1, gauge, ix, mu, 4);
    read_gpu<double>(0, &v2, gauge, iy, nu, 4);
    read_gpu<double>(0, &v3, gauge, iz, mu, 4);
    read_gpu<double>(0, &v4, gauge, ix, nu, 4);

    _suNg_times_suNg(w1, v1, v2);
    _suNg_times_suNg(w2, v4, v3);
    _suNg_times_suNg_dagger(w3, w1, w2);

    _suNg_trace_re(tmpre, w3);

#ifndef GAUGE_SON
    _suNg_trace_im(tmpim, w3);
#endif

    double *t = (double *)res;
    t[0] = tmpre;
    t[1] = tmpim;

#ifdef PLAQ_WEIGHTS
    if (plaq_weight != NULL) {
        t[0] *= plaq_weight[ix * 16 + mu * 4 + nu];
        t[1] *= plaq_weight[ix * 16 + mu * 4 + nu];
    }
#endif
}

__global__ void _avr_plaquette(suNg *u, double *resField, int *iup_gpu, int N, int block_start PLAQ_WEIGHT_ARG_DEF) {
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < N; id += blockDim.x * gridDim.x) {
        const int ix = id + block_start;
        resField[id] = plaq_dev(ix, 1, 0, u, iup_gpu PLAQ_WEIGHT_ARG);
        resField[id] += plaq_dev(ix, 2, 0, u, iup_gpu PLAQ_WEIGHT_ARG);
        resField[id] += plaq_dev(ix, 2, 1, u, iup_gpu PLAQ_WEIGHT_ARG);
        resField[id] += plaq_dev(ix, 3, 0, u, iup_gpu PLAQ_WEIGHT_ARG);
        resField[id] += plaq_dev(ix, 3, 1, u, iup_gpu PLAQ_WEIGHT_ARG);
        resField[id] += plaq_dev(ix, 3, 2, u, iup_gpu PLAQ_WEIGHT_ARG);
    }
}

__global__ void _avr_plaquette_time(suNg *g, double *resPiece, int zero, int global_T, int *iup_gpu, int *timeslices, int N,
                                    int T, int block_start PLAQ_WEIGHT_ARG_DEF) {
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < N; id += blockDim.x * gridDim.x) {
        const int ix = id + block_start;
        int nt = timeslices[ix];
        const int tc = (zero + nt + global_T) % global_T;
        resPiece[id + N * tc] += plaq_dev(ix, 1, 0, g, iup_gpu PLAQ_WEIGHT_ARG);
        resPiece[id + N * tc] += plaq_dev(ix, 2, 0, g, iup_gpu PLAQ_WEIGHT_ARG);
        resPiece[id + N * tc] += plaq_dev(ix, 3, 0, g, iup_gpu PLAQ_WEIGHT_ARG);
        resPiece[id + N * tc + N * global_T] += plaq_dev(ix, 2, 1, g, iup_gpu PLAQ_WEIGHT_ARG);
        resPiece[id + N * tc + N * global_T] += plaq_dev(ix, 3, 1, g, iup_gpu PLAQ_WEIGHT_ARG);
        resPiece[id + N * tc + N * global_T] += plaq_dev(ix, 3, 2, g, iup_gpu PLAQ_WEIGHT_ARG);
    }
}

__global__ void _full_plaquette(suNg *u, hr_complex *resPiece, int *iup_gpu, int N, int block_start PLAQ_WEIGHT_ARG_DEF) {
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < N; id += blockDim.x * gridDim.x) {
        const int ix = id + block_start;
        hr_complex tmp;
        cplaq_dev(&tmp, ix, 1, 0, u, iup_gpu PLAQ_WEIGHT_ARG);
        resPiece[id] = tmp;
        cplaq_dev(&tmp, ix, 2, 0, u, iup_gpu PLAQ_WEIGHT_ARG);
        resPiece[id + N] = tmp;
        cplaq_dev(&tmp, ix, 2, 1, u, iup_gpu PLAQ_WEIGHT_ARG);
        resPiece[id + 2 * N] = tmp;
        cplaq_dev(&tmp, ix, 3, 0, u, iup_gpu PLAQ_WEIGHT_ARG);
        resPiece[id + 3 * N] = tmp;
        cplaq_dev(&tmp, ix, 3, 1, u, iup_gpu PLAQ_WEIGHT_ARG);
        resPiece[id + 4 * N] = tmp;
        cplaq_dev(&tmp, ix, 3, 2, u, iup_gpu PLAQ_WEIGHT_ARG);
        resPiece[id + 5 * N] = tmp;
    }
}

__device__ static void _clover_F_gpu(suNg_algebra_vector *F, suNg *V, int ix, int mu, int nu, int *iup_gpu, int *idn_gpu) {
    int iy, iz, iw;
    suNg v1, v2, v3, v4, w1, w2, w3;

    _suNg_unit(w3);
    _suNg_mul(w3, -4., w3);

    iy = iup_gpu[4 * ix + mu];
    iz = iup_gpu[4 * ix + nu];

    read_gpu<double>(0, &v1, V, ix, mu, 4);
    read_gpu<double>(0, &v2, V, iy, nu, 4);
    read_gpu<double>(0, &v3, V, iz, mu, 4);
    read_gpu<double>(0, &v4, V, ix, nu, 4);

    _suNg_times_suNg(w1, v1, v2);
    _suNg_times_suNg_dagger(w2, w1, v3);
    _suNg_times_suNg_dagger(w1, w2, v4);
    _suNg_add_assign(w3, w1);

    iy = idn_gpu[4 * ix + mu];
    iz = iup_gpu[4 * iy + nu];

    read_gpu<double>(0, &v1, V, ix, nu, 4);
    read_gpu<double>(0, &v2, V, iz, mu, 4);
    read_gpu<double>(0, &v3, V, iy, nu, 4);
    read_gpu<double>(0, &v4, V, iy, mu, 4);

    _suNg_times_suNg_dagger(w1, v1, v2);
    _suNg_times_suNg_dagger(w2, w1, v3);
    _suNg_times_suNg(w1, w2, v4);
    _suNg_add_assign(w3, w1);

    iy = idn_gpu[4 * ix + mu];
    iz = idn_gpu[4 * iy + nu];
    iw = idn_gpu[4 * ix + nu];

    read_gpu<double>(0, &v1, V, iy, mu, 4);
    read_gpu<double>(0, &v2, V, iz, nu, 4);
    read_gpu<double>(0, &v3, V, iz, mu, 4);
    read_gpu<double>(0, &v4, V, iw, nu, 4);

    _suNg_times_suNg(w1, v2, v1);
    _suNg_dagger_times_suNg(w2, w1, v3);
    _suNg_times_suNg(w1, w2, v4);
    _suNg_add_assign(w3, w1);

    iy = idn_gpu[4 * ix + nu];
    iz = iup_gpu[4 * iy + mu];

    read_gpu<double>(0, &v1, V, iy, nu, 4);
    read_gpu<double>(0, &v2, V, iy, mu, 4);
    read_gpu<double>(0, &v3, V, iz, nu, 4);
    read_gpu<double>(0, &v4, V, ix, mu, 4);

    _suNg_dagger_times_suNg(w1, v1, v2);
    _suNg_times_suNg(w2, w1, v3);
    _suNg_times_suNg_dagger(w1, w2, v4);
    _suNg_add_assign(w3, w1);

    _fund_algebra_project(*F, w3);
    _algebra_vector_mul_g(*F, 1 / 4., *F);
}

__global__ void _E_gpu(suNg *v, double *resField, int *iup_gpu, int N, int block_start) {
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < N; id += blockDim.x * gridDim.x) {
        const int ix = id + block_start;
        double p = 0.0;
        resField[ix] = 0.0;
        for (int mu = 0; mu < 4; mu++) {
            for (int nu = mu + 1; nu < 4; nu++) {
                p = plaq_dev(ix, mu, nu, v, iup_gpu);
                resField[ix] += NG - p;
            }
        }
    }
}

__global__ void _E_T_gpu(suNg *v, double *resField, int *iup_gpu, int *timeslices, int zero, int global_T, int N,
                         int block_start) {
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < N; id += blockDim.x * gridDim.x) {
        const int ix = id + block_start;
        int nt = timeslices[ix];
        const int tc = (zero + nt + global_T) % global_T;
        double p = 0.0;
        int mu = 0;
        for (int nu = 1; nu < 4; nu++) {
            p = plaq_dev(ix, mu, nu, v, iup_gpu);
            resField[id + N * tc] += NG - p;
        }
        for (mu = 1; mu < 3; mu++) {
            for (int nu = mu + 1; nu < 4; nu++) {
                p = plaq_dev(ix, mu, nu, v, iup_gpu);
                resField[id + 2 * N * tc] += NG - p;
            }
        }
    }
}

__global__ void _Esym_gpu(suNg *V, double *resField, int *iup_gpu, int *idn_gpu, int N, int block_start) {
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < N; id += blockDim.x * gridDim.x) {
        const int ix = id + block_start;
        suNg_algebra_vector clover;
        double p;
        resField[ix] = 0.0;
        for (int mu = 0; mu < 4; mu++) {
            for (int nu = mu + 1; nu < 4; nu++) {
                _clover_F_gpu(&clover, V, ix, mu, nu, iup_gpu, idn_gpu);
                _algebra_vector_sqnorm_g(p, clover);
                resField[ix] += p;
            }
        }
    }
}

__global__ void _Esym_T_gpu(suNg *v, double *resField, int *iup_gpu, int *idn_gpu, int *timeslices, int zero, int global_T,
                            int N, int block_start) {
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < N; id += blockDim.x * gridDim.x) {
        const int ix = id + block_start;
        int nt = timeslices[ix];
        const int tc = (zero + nt + global_T) % global_T;
        suNg_algebra_vector clover;
        double p = 0.0;
        int mu = 0;
        for (int nu = 1; nu < 4; nu++) {
            _clover_F_gpu(&clover, v, ix, mu, nu, iup_gpu, idn_gpu);
            _algebra_vector_sqnorm_g(p, clover);
            resField[id + N * tc] += p;
        }
        for (mu = 1; mu < 3; mu++) {
            for (int nu = mu + 1; nu < 4; nu++) {
                _clover_F_gpu(&clover, v, ix, mu, nu, iup_gpu, idn_gpu);
                _algebra_vector_sqnorm_g(p, clover);
                resField[id + 2 * N * tc] += p;
            }
        }
    }
}

__global__ void _topo_gpu(double *resField, suNg *v, int *iup_gpu, int *idn_gpu, int N, int block_start) {
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < N; id += blockDim.x * gridDim.x) {
        const int ix = id + block_start;
        resField[ix] = 0.0;
        suNg_algebra_vector F1, F2;
        _clover_F_gpu(&F1, v, ix, 1, 2, iup_gpu, idn_gpu);
        _clover_F_gpu(&F2, v, ix, 0, 3, iup_gpu, idn_gpu);
        for (int i = 0; i < NG * NG - 1; i++) {
            resField[ix] += F1.c[i] * F2.c[i];
        }

        _clover_F_gpu(&F1, v, ix, 1, 3, iup_gpu, idn_gpu);
        _clover_F_gpu(&F2, v, ix, 0, 2, iup_gpu, idn_gpu);
        for (int i = 0; i < NG * NG - 1; i++) {
            resField[ix] -= F1.c[i] * F2.c[i];
        }

        _clover_F_gpu(&F1, v, ix, 0, 1, iup_gpu, idn_gpu);
        _clover_F_gpu(&F2, v, ix, 2, 3, iup_gpu, idn_gpu);
        for (int i = 0; i < NG * NG - 1; i++) {
            resField[ix] += F1.c[i] * F2.c[i];
        }
    }
}