#include "geometry.h"
#include "libhr_core.h"
#include "update.h"
#include "utils.h"

#define _PROJ_BIT (1 << 4)

visible __forceinline__ void suNg_Exp_gpu(suNg *u, suNg *Xin) {
    suNg_algebra_vector h, v;

    h.c[0] = cimag(Xin->c[1]);
    h.c[1] = creal(Xin->c[1]);
    h.c[2] = cimag(Xin->c[0]);

    double z = sqrt(h.c[0] * h.c[0] + h.c[1] * h.c[1] + h.c[2] * h.c[2]);
    double s = 1.;
    if (z > 1e-16) { s = sin(z) / z; }
    double c = cos(z);
    v.c[0] = h.c[0] * s;
    v.c[1] = h.c[1] * s;
    v.c[2] = h.c[2] * s;

    u->c[0] = c + I * v.c[2];
    u->c[1] = v.c[1] + I * v.c[0];
    u->c[2] = -v.c[1] + I * v.c[0];
    u->c[3] = c + I * -v.c[2];
}

visible __forceinline__ void ExpX_gpu(double dt, suNg_algebra_vector *h, suNg *u) {
#ifdef WITH_QUATERNIONS
    suNg v_tmp, u_tmp;

    u_tmp = *u;
    _suNg_exp(dt, *h, v_tmp);
    _suNg_times_suNg(*u, v_tmp, u_tmp);
#else //WITH_QUATERNIONS
    suNg tmp1, tmp2;

    _fund_algebra_represent(tmp1, *h);
    _suNg_mul(tmp1, dt, tmp1);

    suNg_Exp_gpu(&tmp2, &tmp1);
    tmp1 = *u;

    _suNg_times_suNg(*u, tmp2, tmp1);

#endif //WITH_QUATERNIONS
}

__global__ void field_update_kernel(suNg *suNg_field, suNg_algebra_vector *force, int N, int block_start, double dt) {
    suNg_algebra_vector f;
    suNg u;
    for (int id = blockIdx.x * blockDim.x + threadIdx.x; id < N; id += gridDim.x * blockDim.x) {
        const int ix = id + block_start;
        for (int comp = 0; comp < 4; comp++) {
            read_gpu<double>(0, &u, suNg_field, ix, comp, 4);
            read_gpu<double>(0, &f, force, ix, comp, 4);
            ExpX_gpu(dt, &f, &u);
            write_gpu<double>(0, &u, suNg_field, ix, comp, 4);
            write_gpu<double>(0, &f, force, ix, comp, 4);
        }
    }
}

void exec_field_update(suNg_field *suNg_field, suNg_av_field *force, double dt) {
    _PIECE_FOR(&glattice, ixp) {
        const int N = glattice.master_end[ixp] - glattice.master_start[ixp] + 1;
        const int block_start = glattice.master_start[ixp];
        const int grid = (N - 1) / BLOCK_SIZE + 1;
        field_update_kernel<<<grid, BLOCK_SIZE, 0, 0>>>(suNg_field->gpu_ptr, force->gpu_ptr, N, block_start, dt);
    }
}
