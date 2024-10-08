/***************************************************************************\
* Copyright (c) 2024, Sofie Martins                                         *
* All rights reserved.                                                      *
\***************************************************************************/

// Ported from boundary_conditions_core.c by Agostino Patella and
// Claudio Pica

/***************************************************************************/
/* BOUNDARY CONDITIONS TO BE APPLIED ON THE CLOVER TERM                    */
/***************************************************************************/

#ifdef WITH_GPU

#include "libhr_core.h"
#include "./boundary_conditions_gpu_kernels.hpp"

extern "C" {

#if (defined(WITH_EXPCLOVER) || defined(WITH_CLOVER)) && (defined(BC_T_SF) || defined(BC_T_SF_ROTATED))
void cl_SF_BCs_gpu(clover_term *cl) {
    if (COORD[0] == 0) {
        int block_dim = 4;
        int grid_x = (X_EXT - 1) / block_dim + 1;
        int grid_y = (Y_EXT - 1) / block_dim + 1;
        int grid_z = (Z_EXT - 1) / block_dim + 1;
        dim3 block(block_dim, block_dim, block_dim);
        // validation check on these parameters TODO
        dim3 grid(grid_x, grid_y, grid_z);
        apply_cl_SF_BCs<<<grid, block, 0, 0>>>(cl->gpu_ptr, T_BORDER, ipt_gpu, X_EXT, Y_EXT, Z_EXT);
        CudaCheckError();
    }
}

void (*cl_SF_BCs)(clover_term *cl) = cl_SF_BCs_gpu;
#endif

#if (defined(WITH_EXPCLOVER) || defined(WITH_CLOVER)) && defined(BC_T_OPEN)
void cl_open_BCs_gpu(clover_term *cl) {
    if (COORD[0] == 0) {
        int block_dim = 4;
        int grid_x = (X_EXT - 1) / block_dim + 1;
        int grid_y = (Y_EXT - 1) / block_dim + 1;
        int grid_z = (Z_EXT - 1) / block_dim + 1;
        dim3 block(block_dim, block_dim, block_dim);
        dim3 grid(grid_x, grid_y, grid_z);
        apply_cl_open_BCs1<<<grid, block, 0, 0>>>(cl->gpu_ptr, T_BORDER, ipt_gpu, X_EXT, Y_EXT, Z_EXT);
        CudaCheckError();
    }

    if (COORD[0] == NP_T - 1) {
        int block_dim = 4;
        int grid_x = (X_EXT - 1) / block_dim + 1;
        int grid_y = (Y_EXT - 1) / block_dim + 1;
        int grid_z = (Z_EXT - 1) / block_dim + 1;
        dim3 block(block_dim, block_dim, block_dim);
        dim3 grid(grid_x, grid_y, grid_z);
        apply_cl_open_BCs2<<<grid, block, 0, 0>>>(cl->gpu_ptr, T_BORDER, ipt_gpu, X_EXT, Y_EXT, Z_EXT);
        CudaCheckError();
    }
}

void (*cl_open_BCs)(clover_term *cl) = cl_open_BCs_gpu;
#endif

/***************************************************************************/
/* BOUNDARY CONDITIONS TO BE APPLIED ON THE REPRESENTED GAUGE FIELD        */
/***************************************************************************/

#ifdef BC_T_ANTIPERIODIC
void sp_T_antiperiodic_BCs_gpu() {
    if (COORD[0] == 0) {
        int block_dim = 4;
        int grid_x = (X_EXT - 1) / block_dim + 1;
        int grid_y = (Y_EXT - 1) / block_dim + 1;
        int grid_z = (Z_EXT - 1) / block_dim + 1;
        dim3 block(block_dim, block_dim, block_dim);
        dim3 grid(grid_x, grid_y, grid_z);
        apply_boundary_conditions_T<<<grid, block, 0, 0>>>(u_gauge_f->gpu_ptr, T_BORDER, ipt_gpu, X_EXT, Y_EXT, Z_EXT);
        CudaCheckError();
    }
}

void (*sp_T_antiperiodic_BCs)() = sp_T_antiperiodic_BCs_gpu;
#endif

#ifdef BC_X_ANTIPERIODIC
void sp_X_antiperiodic_BCs_gpu() {
    if (COORD[1] == 0) {
        int block_dim = 4;
        int grid_t = (T_EXT - 1) / block_dim + 1;
        int grid_y = (Y_EXT - 1) / block_dim + 1;
        int grid_z = (Z_EXT - 1) / block_dim + 1;
        dim3 block(block_dim, block_dim, block_dim);
        dim3 grid(grid_t, grid_y, grid_z);
        apply_boundary_conditions_X<<<grid, block, 0, 0>>>(u_gauge_f->gpu_ptr, X_BORDER, ipt_gpu, T_EXT, Y_EXT, Z_EXT);
        CudaCheckError();
    }
}

void (*sp_X_antiperiodic_BCs)() = sp_X_antiperiodic_BCs_gpu;
#endif

#ifdef BC_Y_ANTIPERIODIC
void sp_Y_antiperiodic_BCs_gpu() {
    if (COORD[2] == 0) {
        int block_dim = 4;
        int grid_t = (T_EXT - 1) / block_dim + 1;
        int grid_x = (X_EXT - 1) / block_dim + 1;
        int grid_z = (Z_EXT - 1) / block_dim + 1;
        dim3 block(block_dim, block_dim, block_dim);
        dim3 grid(grid_t, grid_x, grid_z);
        apply_boundary_conditions_Y<<<grid, block, 0, 0>>>(u_gauge_f->gpu_ptr, Y_BORDER, ipt_gpu, T_EXT, X_EXT, Z_EXT);
        CudaCheckError();
    }
}

void (*sp_Y_antiperiodic_BCs)() = sp_Y_antiperiodic_BCs_gpu;
#endif

#ifdef BC_Z_ANTIPERIODIC
void sp_Z_antiperiodic_BCs_gpu() {
    if (COORD[3] == 0) {
        int block_dim = 4;
        int grid_t = (T_EXT - 1) / block_dim + 1;
        int grid_x = (X_EXT - 1) / block_dim + 1;
        int grid_y = (Y_EXT - 1) / block_dim + 1;
        dim3 block(block_dim, block_dim, block_dim);
        dim3 grid(grid_t, grid_x, grid_y);
        apply_boundary_conditions_Z<<<grid, block, 0, 0>>>(u_gauge_f->gpu_ptr, Z_BORDER, ipt_gpu, T_EXT, X_EXT, Y_EXT);
        CudaCheckError();
    }
}

void (*sp_Z_antiperiodic_BCs)() = sp_Z_antiperiodic_BCs_gpu;
#endif

#ifdef BC_T_SF_ROTATED
void chiSF_ds_BT_gpu(double ds) {
    if (COORD[0] == 0) {
        int block_dim = 4;
        int grid_x = (X_EXT - 1) / block_dim + 1;
        int grid_y = (Y_EXT - 1) / block_dim + 1;
        int grid_z = (Z_EXT - 1) / block_dim + 1;
        dim3 block(block_dim, block_dim, block_dim);
        dim3 grid(grid_x, grid_y, grid_z);
        apply_chiSF_ds_BT<<<grid, block, 0, 0>>>(ds, u_gauge_f->gpu_ptr, T_BORDER + 1, ipt_gpu, X_EXT, Y_EXT, Z_EXT);
        CudaCheckError();
    }

    if (COORD[0] == NP_T - 1) {
        int block_dim = 4;
        int grid_x = (X_EXT - 1) / block_dim + 1;
        int grid_y = (Y_EXT - 1) / block_dim + 1;
        int grid_z = (Z_EXT - 1) / block_dim + 1;
        dim3 block(block_dim, block_dim, block_dim);
        dim3 grid(grid_x, grid_y, grid_z);
        apply_chiSF_ds_BT<<<grid, block, 0, 0>>>(ds, u_gauge_f->gpu_ptr, T + T_BORDER - 1, ipt_gpu, X_EXT, Y_EXT, Z_EXT);
        CudaCheckError();
    }
}

void (*chiSF_ds_BT)(double ds) = chiSF_ds_BT_gpu;
#endif

/***************************************************************************/
/* BOUNDARY CONDITIONS TO BE APPLIED ON THE FUNDAMENTAL GAUGE FIELD        */
/***************************************************************************/

#if defined(BC_T_SF) || defined(BC_T_SF_ROTATED)
void gf_SF_BCs_gpu(suNg *dn_h, suNg *up_h) {
    suNg *dn, *up;
    cudaMalloc((void **)&dn, sizeof(suNg));
    cudaMalloc((void **)&up, sizeof(suNg));
    cudaMemcpy(dn, dn_h, sizeof(suNg), cudaMemcpyHostToDevice);
    cudaMemcpy(up, up_h, sizeof(suNg), cudaMemcpyHostToDevice);
    if (COORD[0] == 0) {
        int block_dim = 4;
        int grid_x = (X_EXT - 1) / block_dim + 1;
        int grid_y = (Y_EXT - 1) / block_dim + 1;
        int grid_z = (Z_EXT - 1) / block_dim + 1;
        dim3 block(block_dim, block_dim, block_dim);
        dim3 grid(grid_x, grid_y, grid_z);
        apply_gf_SF_BCs_1<<<grid, block, 0, 0>>>(u_gauge->gpu_ptr, up, dn, T_BORDER, ipt_gpu, X_EXT, Y_EXT, Z_EXT);
        CudaCheckError();
    }

    if (COORD[0] == NP_T - 1) {
        int block_dim = 4;
        int grid_x = (X_EXT - 1) / block_dim + 1;
        int grid_y = (Y_EXT - 1) / block_dim + 1;
        int grid_z = (Z_EXT - 1) / block_dim + 1;
        dim3 block(block_dim, block_dim, block_dim);
        dim3 grid(grid_x, grid_y, grid_z);
        apply_gf_SF_BCs_2<<<grid, block, 0, 0>>>(u_gauge->gpu_ptr, up, dn, T_BORDER, ipt_gpu, X_EXT, Y_EXT, Z_EXT);
        CudaCheckError();
    }
}

void (*gf_SF_BCs)(suNg *dn, suNg *up) = gf_SF_BCs_gpu;
#endif

#if defined(BC_T_SF) || defined(BC_T_SF_ROTATED)
void SF_classical_solution_core_gpu(suNg *U_h, int it) {
    suNg U;
    cudaMemcpy(&U, U_h, sizeof(suNg), cudaMemcpyHostToDevice);
    if (COORD[0] == 0) {
        int block_dim = 4;
        int grid_x = (X_EXT - 1) / block_dim + 1;
        int grid_y = (Y_EXT - 1) / block_dim + 1;
        int grid_z = (Z_EXT - 1) / block_dim + 1;
        dim3 block(block_dim, block_dim, block_dim);
        dim3 grid(grid_x, grid_y, grid_z);
        apply_SF_classical_solution<<<grid, block, 0, 0>>>(u_gauge->gpu_ptr, &U, it, T_BORDER, ipt_gpu, X_EXT, Y_EXT, Z_EXT);
        CudaCheckError();
    }
}

void (*SF_classical_solution_core)(suNg *U, int it) = SF_classical_solution_core_gpu;
#endif

#ifdef BC_T_OPEN
void gf_open_BCs_gpu() {
    if (COORD[0] == 0) {
        if (T_BORDER > 0) {
            int block_dim = 4;
            int grid_x = (X_EXT - 1) / block_dim + 1;
            int grid_y = (Y_EXT - 1) / block_dim + 1;
            int grid_z = (Z_EXT - 1) / block_dim + 1;
            dim3 block(block_dim, block_dim, block_dim);
            dim3 grid(grid_x, grid_y, grid_z);
            apply_gf_open_BCs<<<grid, block, 0, 0>>>(u_gauge->gpu_ptr, T_BORDER - 1, ipt_gpu, X_EXT, Y_EXT, Z_EXT);
            CudaCheckError();
        }
    }

    if (COORD[0] == NP_T - 1) {
        do {
            int block_dim = 4;
            int grid_x = (X_EXT - 1) / block_dim + 1;
            int grid_y = (Y_EXT - 1) / block_dim + 1;
            int grid_z = (Z_EXT - 1) / block_dim + 1;
            dim3 block(block_dim, block_dim, block_dim);
            dim3 grid(grid_x, grid_y, grid_z);
            apply_gf_open_BCs2<<<grid, block, 0, 0>>>(u_gauge->gpu_ptr, T + T_BORDER - 1, ipt_gpu, X_EXT, Y_EXT, Z_EXT);
            CudaCheckError();
        } while (0);

        if (T_BORDER > 0) {
            int block_dim = 4;
            int grid_x = (X_EXT - 1) / block_dim + 1;
            int grid_y = (Y_EXT - 1) / block_dim + 1;
            int grid_z = (Z_EXT - 1) / block_dim + 1;
            dim3 block(block_dim, block_dim, block_dim);
            dim3 grid(grid_x, grid_y, grid_z);
            apply_gf_open_BCs<<<grid, block, 0, 0>>>(u_gauge->gpu_ptr, T + T_BORDER, ipt_gpu, X_EXT, Y_EXT, Z_EXT);
            CudaCheckError();
        }
    }
}

void (*gf_open_BCs)() = gf_open_BCs_gpu;
#endif

/***************************************************************************/
/* BOUNDARY CONDITIONS TO BE APPLIED ON THE MOMENTUM FIELDS                */
/***************************************************************************/

#if defined(BC_T_SF) || defined(BC_T_SF_ROTATED)
void mf_Dirichlet_BCs_gpu(suNg_av_field *force) {
    if (COORD[0] == 0) {
        if (T_BORDER > 0) {
            int block_dim = 4;
            int grid_x = (X_EXT - 1) / block_dim + 1;
            int grid_y = (Y_EXT - 1) / block_dim + 1;
            int grid_z = (Z_EXT - 1) / block_dim + 1;
            dim3 block(block_dim, block_dim, block_dim);
            dim3 grid(grid_x, grid_y, grid_z);
            apply_mf_Dirichlet_BCs<<<grid, block, 0, 0>>>(force->gpu_ptr, T_BORDER - 1, ipt_gpu, X_EXT, Y_EXT, Z_EXT);
            CudaCheckError();
        }

        do {
            int block_dim = 4;
            int grid_x = (X_EXT - 1) / block_dim + 1;
            int grid_y = (Y_EXT - 1) / block_dim + 1;
            int grid_z = (Z_EXT - 1) / block_dim + 1;
            dim3 block(block_dim, block_dim, block_dim);
            dim3 grid(grid_x, grid_y, grid_z);
            apply_mf_Dirichlet_BCs<<<grid, block, 0, 0>>>(force->gpu_ptr, T_BORDER, ipt_gpu, X_EXT, Y_EXT, Z_EXT);
            CudaCheckError();
            apply_mf_Dirichlet_BCs_spatial<<<grid, block, 0, 0>>>(force->gpu_ptr, T_BORDER + 1, ipt_gpu, X_EXT, Y_EXT, Z_EXT);
            CudaCheckError();
        } while (0);
    }

    if (COORD[0] == NP_T - 1) {
        do {
            int block_dim = 4;
            int grid_x = (X_EXT - 1) / block_dim + 1;
            int grid_y = (Y_EXT - 1) / block_dim + 1;
            int grid_z = (Z_EXT - 1) / block_dim + 1;
            dim3 block(block_dim, block_dim, block_dim);
            dim3 grid(grid_x, grid_y, grid_z);
            apply_mf_Dirichlet_BCs<<<grid, block, 0, 0>>>(force->gpu_ptr, T + T_BORDER - 1, ipt_gpu, X_EXT, Y_EXT, Z_EXT);
            CudaCheckError();
        } while (0);

        if (T_BORDER > 0) {
            int block_dim = 4;
            int grid_x = (X_EXT - 1) / block_dim + 1;
            int grid_y = (Y_EXT - 1) / block_dim + 1;
            int grid_z = (Z_EXT - 1) / block_dim + 1;
            dim3 block(block_dim, block_dim, block_dim);
            dim3 grid(grid_x, grid_y, grid_z);
            apply_mf_Dirichlet_BCs<<<grid, block, 0, 0>>>(force->gpu_ptr, T + T_BORDER, ipt_gpu, X_EXT, Y_EXT, Z_EXT);
            CudaCheckError();
        }
    }
}

void (*mf_Dirichlet_BCs)(suNg_av_field *force) = mf_Dirichlet_BCs_gpu;
#endif

#ifdef BC_T_OPEN
void mf_open_BCs_gpu(suNg_av_field *force) {
    if (COORD[0] == 0) {
        if (T_BORDER > 0) {
            int block_dim = 4;
            int grid_x = (X_EXT - 1) / block_dim + 1;
            int grid_y = (Y_EXT - 1) / block_dim + 1;
            int grid_z = (Z_EXT - 1) / block_dim + 1;
            dim3 block(block_dim, block_dim, block_dim);
            dim3 grid(grid_x, grid_y, grid_z);
            apply_mf_Dirichlet_BCs<<<grid, block, 0, 0>>>(force->gpu_ptr, T_BORDER - 1, ipt_gpu, X_EXT, Y_EXT, Z_EXT);
            CudaCheckError();
        }
    }

    if (COORD[0] == NP_T - 1) {
        do {
            int block_dim = 4;
            int grid_x = (X_EXT - 1) / block_dim + 1;
            int grid_y = (Y_EXT - 1) / block_dim + 1;
            int grid_z = (Z_EXT - 1) / block_dim + 1;
            dim3 block(block_dim, block_dim, block_dim);
            dim3 grid(grid_x, grid_y, grid_z);
            apply_mf_Dirichlet_BCs_temporal<<<grid, block, 0, 0>>>(force->gpu_ptr, T + T_BORDER - 1, ipt_gpu, X_EXT, Y_EXT,
                                                                   Z_EXT);
            CudaCheckError();
        } while (0);

        if (T_BORDER > 0) {
            int block_dim = 4;
            int grid_x = (X_EXT - 1) / block_dim + 1;
            int grid_y = (Y_EXT - 1) / block_dim + 1;
            int grid_z = (Z_EXT - 1) / block_dim + 1;
            dim3 block(block_dim, block_dim, block_dim);
            dim3 grid(grid_x, grid_y, grid_z);
            apply_mf_Dirichlet_BCs<<<grid, block, 0, 0>>>(force->gpu_ptr, T + T_BORDER, ipt_gpu, X_EXT, Y_EXT, Z_EXT);
            CudaCheckError();
        }
    }
}

void (*mf_open_BCs)(suNg_av_field *force) = mf_open_BCs_gpu;
#endif

/***************************************************************************/
/* BOUNDARY CONDITIONS TO BE APPLIED ON THE SPINOR FIELDS                  */
/***************************************************************************/

#if defined(BC_T_SF)
void sf_Dirichlet_BCs_gpu(spinor_field *sp) {
    if (COORD[0] == 0) {
        int block_dim = 4;
        int grid_x = (X_EXT - 1) / block_dim + 1;
        int grid_y = (Y_EXT - 1) / block_dim + 1;
        int grid_z = (Z_EXT - 1) / block_dim + 1;
        dim3 block(block_dim, block_dim, block_dim);
        dim3 grid(grid_x, grid_y, grid_z);
        apply_sf_Dirichlet_BCs1<<<grid, block, 0, 0>>>(_GPU_FIELD_BLK(sp, 0), sp->type->master_shift, sp->type->gsize_spinor,
                                                       T_BORDER, ipt_gpu, X_EXT, Y_EXT, Z_EXT);
        CudaCheckError();
    }

    if (COORD[0] == NP_T - 1) {
        int block_dim = 4;
        int grid_x = (X_EXT - 1) / block_dim + 1;
        int grid_y = (Y_EXT - 1) / block_dim + 1;
        int grid_z = (Z_EXT - 1) / block_dim + 1;
        dim3 block(block_dim, block_dim, block_dim);
        dim3 grid(grid_x, grid_y, grid_z);
        apply_sf_Dirichlet_BCs2<<<grid, block, 0, 0>>>(_GPU_FIELD_BLK(sp, 0), sp->type->master_shift, sp->type->gsize_spinor,
                                                       T + T_BORDER - 1, ipt_gpu, X_EXT, Y_EXT, Z_EXT);
        CudaCheckError();
    }
}

void (*sf_Dirichlet_BCs)(spinor_field *sp) = sf_Dirichlet_BCs_gpu;
#endif

#if defined(BC_T_SF)
void sf_Dirichlet_BCs_flt_gpu(spinor_field_flt *sp) {
    if (COORD[0] == 0) {
        int block_dim = 4;
        int grid_x = (X_EXT - 1) / block_dim + 1;
        int grid_y = (Y_EXT - 1) / block_dim + 1;
        int grid_z = (Z_EXT - 1) / block_dim + 1;
        dim3 block(block_dim, block_dim, block_dim);
        dim3 grid(grid_x, grid_y, grid_z);
        apply_sf_Dirichlet_BCs1_flt<<<grid, block, 0, 0>>>(_GPU_FIELD_BLK(sp, 0), sp->type->master_shift,
                                                           sp->type->gsize_spinor, T_BORDER, ipt_gpu, X_EXT, Y_EXT, Z_EXT);
        CudaCheckError();
    }

    if (COORD[0] == NP_T - 1) {
        int block_dim = 4;
        int grid_x = (X_EXT - 1) / block_dim + 1;
        int grid_y = (Y_EXT - 1) / block_dim + 1;
        int grid_z = (Z_EXT - 1) / block_dim + 1;
        dim3 block(block_dim, block_dim, block_dim);
        dim3 grid(grid_x, grid_y, grid_z);
        apply_sf_Dirichlet_BCs2_flt<<<grid, block, 0, 0>>>(_GPU_FIELD_BLK(sp, 0), sp->type->master_shift,
                                                           sp->type->gsize_spinor, T + T_BORDER - 1, ipt_gpu, X_EXT, Y_EXT,
                                                           Z_EXT);
        CudaCheckError();
    }
}

void (*sf_Dirichlet_BCs_flt)(spinor_field_flt *sp) = sf_Dirichlet_BCs_flt_gpu;
#endif

#if defined(BC_T_SF_ROTATED)
void sf_open_BCs_gpu(spinor_field *sp) {
    if (COORD[0] == 0) {
        int block_dim = 4;
        int grid_x = (X_EXT - 1) / block_dim + 1;
        int grid_y = (Y_EXT - 1) / block_dim + 1;
        int grid_z = (Z_EXT - 1) / block_dim + 1;
        dim3 block(block_dim, block_dim, block_dim);
        dim3 grid(grid_x, grid_y, grid_z);
        apply_sf_open_BCs<<<grid, block, 0, 0>>>(_GPU_FIELD_BLK(sp, 0), sp->type->master_shift, sp->type->gsize_spinor,
                                                 T_BORDER, ipt_gpu, X_EXT, Y_EXT, Z_EXT);
        CudaCheckError();
    }
}

void (*sf_open_BCs)(spinor_field *sp) = sf_open_BCs_gpu;
#endif

#if defined(BC_T_SF_ROTATED)
void sf_open_BCs_flt_gpu(spinor_field_flt *sp) {
    if (COORD[0] == 0) {
        int block_dim = 4;
        int grid_x = (X_EXT - 1) / block_dim + 1;
        int grid_y = (Y_EXT - 1) / block_dim + 1;
        int grid_z = (Z_EXT - 1) / block_dim + 1;
        dim3 block(block_dim, block_dim, block_dim);
        dim3 grid(grid_x, grid_y, grid_z);
        apply_sf_open_BCs_flt<<<grid, block, 0, 0>>>(_GPU_FIELD_BLK(sp, 0), sp->type->master_shift, sp->type->gsize_spinor,
                                                     T_BORDER, ipt_gpu, X_EXT, Y_EXT, Z_EXT);
        CudaCheckError();
    }
}

void (*sf_open_BCs_flt)(spinor_field_flt *sp) = sf_open_BCs_flt_gpu;
#endif

#ifdef BC_T_OPEN
void sf_open_v2_BCs_gpu(spinor_field *sp) {
    if (COORD[0] == 0) {
        int block_dim = 4;
        int grid_x = (X_EXT - 1) / block_dim + 1;
        int grid_y = (Y_EXT - 1) / block_dim + 1;
        int grid_z = (Z_EXT - 1) / block_dim + 1;
        dim3 block(block_dim, block_dim, block_dim);
        dim3 grid(grid_x, grid_y, grid_z);
        apply_sf_Dirichlet_BCs3<<<grid, block, 0, 0>>>(_GPU_FIELD_BLK(sp, 0), sp->type->master_shift, sp->type->gsize_spinor,
                                                       T_BORDER + 1, ipt_gpu, X_EXT, Y_EXT, Z_EXT);
        CudaCheckError();
        if (T_BORDER > 0) {
            apply_sf_Dirichlet_BCs2<<<grid, block, 0, 0>>>(_GPU_FIELD_BLK(sp, 0), sp->type->master_shift,
                                                           sp->type->gsize_spinor, T_BORDER - 1, ipt_gpu, X_EXT, Y_EXT, Z_EXT);
            CudaCheckError();
        }
    }

    if (COORD[0] == NP_T - 1) {
        int block_dim = 4;
        int grid_x = (X_EXT - 1) / block_dim + 1;
        int grid_y = (Y_EXT - 1) / block_dim + 1;
        int grid_z = (Z_EXT - 1) / block_dim + 1;
        dim3 block(block_dim, block_dim, block_dim);
        dim3 grid(grid_x, grid_y, grid_z);
        apply_sf_Dirichlet_BCs3<<<grid, block, 0, 0>>>(_GPU_FIELD_BLK(sp, 0), sp->type->master_shift, sp->type->gsize_spinor,
                                                       T + T_BORDER, ipt_gpu, X_EXT, Y_EXT, Z_EXT);
        CudaCheckError();
        if (T_BORDER > 0) {
            apply_sf_Dirichlet_BCs2<<<grid, block, 0, 0>>>(_GPU_FIELD_BLK(sp, 0), sp->type->master_shift,
                                                           sp->type->gsize_spinor, T + T_BORDER, ipt_gpu, X_EXT, Y_EXT, Z_EXT);
            CudaCheckError();
        }
    }
}

void (*sf_open_v2_BCs)(spinor_field *sf) = sf_open_v2_BCs_gpu;
#endif

#ifdef BC_T_OPEN
void sf_open_v2_BCs_flt_gpu(spinor_field_flt *sp) {
    if (COORD[0] == 0) {
        int block_dim = 4;
        int grid_x = (X_EXT - 1) / block_dim + 1;
        int grid_y = (Y_EXT - 1) / block_dim + 1;
        int grid_z = (Z_EXT - 1) / block_dim + 1;
        dim3 block(block_dim, block_dim, block_dim);
        dim3 grid(grid_x, grid_y, grid_z);
        apply_sf_Dirichlet_BCs3_flt<<<grid, block, 0, 0>>>(_GPU_FIELD_BLK(sp, 0), sp->type->master_shift,
                                                           sp->type->gsize_spinor, T_BORDER + 1, ipt_gpu, X_EXT, Y_EXT, Z_EXT);
        CudaCheckError();
        if (T_BORDER > 0) {
            apply_sf_Dirichlet_BCs2_flt<<<grid, block, 0, 0>>>(_GPU_FIELD_BLK(sp, 0), sp->type->master_shift,
                                                               sp->type->gsize_spinor, T_BORDER - 1, ipt_gpu, X_EXT, Y_EXT,
                                                               Z_EXT);
            CudaCheckError();
        }
    }

    if (COORD[0] == NP_T - 1) {
        int block_dim = 4;
        int grid_x = (X_EXT - 1) / block_dim + 1;
        int grid_y = (Y_EXT - 1) / block_dim + 1;
        int grid_z = (Z_EXT - 1) / block_dim + 1;
        dim3 block(block_dim, block_dim, block_dim);
        dim3 grid(grid_x, grid_y, grid_z);
        apply_sf_Dirichlet_BCs3_flt<<<grid, block, 0, 0>>>(_GPU_FIELD_BLK(sp, 0), sp->type->master_shift,
                                                           sp->type->gsize_spinor, T + T_BORDER, ipt_gpu, X_EXT, Y_EXT, Z_EXT);
        if (T_BORDER > 0) {
            apply_sf_Dirichlet_BCs2_flt<<<grid, block, 0, 0>>>(_GPU_FIELD_BLK(sp, 0), sp->type->master_shift,
                                                               sp->type->gsize_spinor, T + T_BORDER, ipt_gpu, X_EXT, Y_EXT,
                                                               Z_EXT);
        }
        CudaCheckError();
    }
}

void (*sf_open_v2_BCs_flt)(spinor_field_flt *sp) = sf_open_v2_BCs_flt_gpu;
#endif
}

#endif
