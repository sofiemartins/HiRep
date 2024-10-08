/***************************************************************************\
* Copyright (c) 2008-2024, Agostino Patella, Claudio Pica, Sofie Martins    *
* All rights reserved.                                                      *
\***************************************************************************/

#include "utils.h"
#include "libhr_core.h"
#include "utils.h"
#include "memory.h"
#include "io.h"
#include <math.h>

static int init = 0;
static BCs_pars_t BCs_pars;

static void print_info_bcs() {
    lprintf("BCS", 0,
            "Gauge field: "
#if defined(BC_T_OPEN)
            "OPEN"
#elif defined(BC_T_SF) || defined(BC_T_SF_ROTATED)
            "DIRICHLET"
#else
            "PERIODIC"
#endif
            " x "
#if defined(GAUGE_SPATIAL_TWIST)
            "TWISTED TWISTED TWISTED"
#else
            "PERIODIC PERIODIC PERIODIC"
#endif
            "\n");

    lprintf("BCS", 0,
            "Fermion fields: "
#if defined(BC_T_SF_ROTATED)
            "OPEN"
#elif defined(BC_T_OPEN) || defined(BC_T_SF)
            "DIRICHLET"
#elif defined(BC_T_ANTIPERIODIC)
            "ANTIPERIODIC"
#elif defined(BC_T_THETA)
            "THETA"
#else
            "PERIODIC"
#endif
            " x "
#if defined(BC_X_ANTIPERIODIC)
            "ANTIPERIODIC "
#elif defined(BC_X_THETA)
            "THETA "
#elif defined(GAUGE_SPATIAL_TWIST)
            "TWISTED "
#else
            "PERIODIC "
#endif
#if defined(BC_Y_ANTIPERIODIC)
            "ANTIPERIODIC "
#elif defined(BC_Y_THETA)
            "THETA "
#elif defined(GAUGE_SPATIAL_TWIST)
            "TWISTED "
#else
            "PERIODIC "
#endif
#if defined(BC_Z_ANTIPERIODIC)
            "ANTIPERIODIC"
#elif defined(BC_Z_THETA)
            "THETA"
#elif defined(GAUGE_SPATIAL_TWIST)
            "TWISTED"
#else
            "PERIODIC"
#endif
            "\n");

#if defined(BC_T_SF_ROTATED)
    lprintf("BCS", 0, "Chirally rotated Schroedinger Functional ds=%e BCs=%d(1=Background, 0=no Background)\n",
            BCs_pars.chiSF_boundary_improvement_ds, BCs_pars.SF_BCs);
#elif defined(BC_T_SF)
    lprintf("BCS", 0, "Basic Schroedinger Functional BCs=%d (1=Background, 0=no Background)\n", BCs_pars.SF_BCs);
#endif
}

static void init_plaq_weights() {
#ifdef PLAQ_WEIGHTS
    plaq_weight = malloc(sizeof(double) * glattice.gsize_gauge * 16);
    rect_weight = malloc(sizeof(double) * glattice.gsize_gauge * 16);
    for (int i = 0; i < 16 * glattice.gsize_gauge; i++) {
        rect_weight[i] = 1.0;
        plaq_weight[i] = 1.0;
    }

#ifdef WITH_GPU
    CHECK_CUDA(cudaMemcpy(plaq_weight_gpu, plaq_weight, 16 * glattice.gsize_gauge * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(rect_weight_gpu, rect_weight, 16 * glattice.gsize_gauge * sizeof(double), cudaMemcpyHostToDevice));
#endif
#endif
}

static void init_BCs_parameters(BCs_pars_t *pars) {
    BCs_pars.fermion_twisting_theta[0] = 0.;
    BCs_pars.fermion_twisting_theta[1] = 0.;
    BCs_pars.fermion_twisting_theta[2] = 0.;
    BCs_pars.fermion_twisting_theta[3] = 0.;
    BCs_pars.gauge_boundary_improvement_cs = 1.;
    BCs_pars.gauge_boundary_improvement_ct = 1.;
    BCs_pars.chiSF_boundary_improvement_ds = 1.;
    BCs_pars.SF_BCs = 0;

    if (pars != NULL) { BCs_pars = *pars; }

#ifdef FERMION_THETA

#ifndef BC_T_THETA
    BCs_pars.fermion_twisting_theta[0] = 0.;
#endif
#ifndef BC_X_THETA
    BCs_pars.fermion_twisting_theta[1] = 0.;
#endif
#ifndef BC_Y_THETA
    BCs_pars.fermion_twisting_theta[2] = 0.;
#endif
#ifndef BC_Z_THETA
    BCs_pars.fermion_twisting_theta[3] = 0.;
#endif

    lprintf("BCS", 0, "Fermion twisting theta angles = %e %e %e %e\n", BCs_pars.fermion_twisting_theta[0],
            BCs_pars.fermion_twisting_theta[1], BCs_pars.fermion_twisting_theta[2], BCs_pars.fermion_twisting_theta[3]);

    eitheta[0] =
        cos(BCs_pars.fermion_twisting_theta[0] / (double)GLB_T) + I * sin(BCs_pars.fermion_twisting_theta[0] / (double)GLB_T);
    eitheta[1] =
        cos(BCs_pars.fermion_twisting_theta[1] / (double)GLB_X) + I * sin(BCs_pars.fermion_twisting_theta[1] / (double)GLB_X);
    eitheta[2] =
        cos(BCs_pars.fermion_twisting_theta[2] / (double)GLB_Y) + I * sin(BCs_pars.fermion_twisting_theta[2] / (double)GLB_Y);
    eitheta[3] =
        cos(BCs_pars.fermion_twisting_theta[3] / (double)GLB_Z) + I * sin(BCs_pars.fermion_twisting_theta[3] / (double)GLB_Z);

#endif /* FERMION_THETA */

#ifdef BC_T_OPEN
    lprintf("BCS", 0, "Open BC gauge boundary term ct=%e cs=%e\n", BCs_pars.gauge_boundary_improvement_ct,
            BCs_pars.gauge_boundary_improvement_cs);
    init_plaq_open_BCs(plaq_weight, rect_weight, BCs_pars.gauge_boundary_improvement_ct,
                       BCs_pars.gauge_boundary_improvement_cs);
#ifdef WITH_GPU
    CHECK_CUDA(cudaMemcpy(plaq_weight_gpu, plaq_weight, 16 * glattice.gsize_gauge * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(rect_weight_gpu, rect_weight, 16 * glattice.gsize_gauge * sizeof(double), cudaMemcpyHostToDevice));
#endif
#endif

#if defined(BC_T_SF) || defined(BC_T_SF_ROTATED)
    lprintf("BCS", 0, "BC gauge boundary term ct=%e\n", BCs_pars.gauge_boundary_improvement_ct);
    if (BCs_pars.SF_BCs == 0) {
        _suNg_unit(BCs_pars.gauge_boundary_dn);
        _suNg_unit(BCs_pars.gauge_boundary_up);
    } else {
        init_gf_SF_BCs(&(BCs_pars.gauge_boundary_dn), &(BCs_pars.gauge_boundary_up));
    }
    init_plaq_SF_BCs(BCs_pars.gauge_boundary_improvement_ct);
#endif
}

void init_BCs(BCs_pars_t *pars) {
    error(init == 1, 1, "init_BCs [boundary_conditions.c]", "BCs already initialized");
    init = 1;
    init_plaq_weights();
    print_info_bcs();
    init_BCs_parameters(pars);

#ifdef GAUGE_SPATIAL_TWIST
    init_plaq_twisted_BCs();
#endif
}

void free_BCs(void) {
    if (init == 0) { return; }
    init = 0;

#ifdef PLAQ_WEIGHTS
    if (plaq_weight != NULL) { free(plaq_weight); }
    if (rect_weight != NULL) { free(rect_weight); }
#ifdef WITH_GPU
    if (plaq_weight_gpu != NULL) { cudaFree(plaq_weight_gpu); }
    if (rect_weight_gpu != NULL) { cudaFree(rect_weight_gpu); }
#endif
#endif
}

#ifdef WITH_GPU
void apply_BCs_on_represented_gauge_field_gpu() {
#ifdef BC_T_ANTIPERIODIC
    sp_T_antiperiodic_BCs_gpu();
#endif
#ifdef BC_X_ANTIPERIODIC
    sp_X_antiperiodic_BCs_gpu();
#endif
#ifdef BC_Y_ANTIPERIODIC
    sp_Y_antiperiodic_BCs_gpu();
#endif
#ifdef BC_Z_ANTIPERIODIC
    sp_Z_antiperiodic_BCs_gpu();
#endif
#ifdef BC_T_SF_ROTATED
#ifndef ALLOCATE_REPR_GAUGE_FIELD
#error The represented gauge field must be allocated!!!
#endif
    chiSF_ds_BT_gpu(BCs_pars.chiSF_boundary_improvement_ds);
#endif
}

void (*apply_BCs_on_represented_gauge_field)() = apply_BCs_on_represented_gauge_field_gpu;
#endif

void apply_BCs_on_represented_gauge_field_cpu() {
#ifdef BC_T_ANTIPERIODIC
    sp_T_antiperiodic_BCs_cpu();
#endif
#ifdef BC_X_ANTIPERIODIC
    sp_X_antiperiodic_BCs_cpu();
#endif
#ifdef BC_Y_ANTIPERIODIC
    sp_Y_antiperiodic_BCs_cpu();
#endif
#ifdef BC_Z_ANTIPERIODIC
    sp_Z_antiperiodic_BCs_cpu();
#endif
#ifdef BC_T_SF_ROTATED
#ifndef ALLOCATE_REPR_GAUGE_FIELD
#error The represented gauge field must be allocated!!!
#endif
    chiSF_ds_BT_cpu(BCs_pars.chiSF_boundary_improvement_ds);
#endif
}

#ifndef WITH_GPU
void (*apply_BCs_on_represented_gauge_field)() = apply_BCs_on_represented_gauge_field_cpu;
#endif

#ifdef WITH_GPU
void apply_BCs_on_fundamental_gauge_field_gpu(void) {
    complete_sendrecv_suNg_field(u_gauge);
#if defined(BC_T_SF) || defined(BC_T_SF_ROTATED)
    gf_SF_BCs_gpu(&BCs_pars.gauge_boundary_dn, &BCs_pars.gauge_boundary_up);
#endif
#ifdef BC_T_OPEN
    gf_open_BCs_gpu();
#endif
}

void (*apply_BCs_on_fundamental_gauge_field)(void) = apply_BCs_on_fundamental_gauge_field_gpu;
#endif

void apply_BCs_on_fundamental_gauge_field_cpu(void) {
    complete_sendrecv_suNg_field(u_gauge);
#if defined(BC_T_SF) || defined(BC_T_SF_ROTATED)
    gf_SF_BCs_cpu(&BCs_pars.gauge_boundary_dn, &BCs_pars.gauge_boundary_up);
#endif
#ifdef BC_T_OPEN
    gf_open_BCs_cpu();
#endif
}

#ifndef WITH_GPU
void (*apply_BCs_on_fundamental_gauge_field)(void) = apply_BCs_on_fundamental_gauge_field_cpu;
#endif

#ifdef WITH_GPU
void apply_BCs_on_momentum_field_gpu(suNg_av_field *force) {
#if defined(BC_T_SF) || defined(BC_T_SF_ROTATED)
    mf_Dirichlet_BCs_gpu(force);
#endif
#ifdef BC_T_OPEN
    mf_open_BCs_gpu(force);
#endif
}

void (*apply_BCs_on_momentum_field)(suNg_av_field *force) = apply_BCs_on_momentum_field_gpu;
#endif

void apply_BCs_on_momentum_field_cpu(suNg_av_field *force) {
#if defined(BC_T_SF) || defined(BC_T_SF_ROTATED)
    mf_Dirichlet_BCs_cpu(force);
#endif
#ifdef BC_T_OPEN
    mf_open_BCs_cpu(force);
#endif
}

#ifndef WITH_GPU
void (*apply_BCs_on_momentum_field)(suNg_av_field *force) = apply_BCs_on_momentum_field_cpu;
#endif

#ifdef WITH_GPU
void apply_BCs_on_spinor_field_gpu(spinor_field *sp) {
#ifdef BC_T_SF
    sf_Dirichlet_BCs_gpu(sp);
#endif
#ifdef BC_T_SF_ROTATED
    sf_open_BCs_gpu(sp);
#endif
#ifdef BC_T_OPEN
    sf_open_v2_BCs_gpu(sp);
#endif
}

void (*apply_BCs_on_spinor_field)(spinor_field *sp) = apply_BCs_on_spinor_field_gpu;
#endif

void apply_BCs_on_spinor_field_cpu(spinor_field *sp) {
#ifdef BC_T_SF
    sf_Dirichlet_BCs_cpu(sp);
#endif
#ifdef BC_T_SF_ROTATED
    sf_open_BCs_cpu(sp);
#endif
#ifdef BC_T_OPEN
    sf_open_v2_BCs_cpu(sp);
#endif
}

#ifndef WITH_GPU
void (*apply_BCs_on_spinor_field)(spinor_field *sp) = apply_BCs_on_spinor_field_cpu;
#endif

#ifdef WITH_GPU
void apply_BCs_on_spinor_field_flt_gpu(spinor_field_flt *sp) {
#if defined(BC_T_SF)
    sf_Dirichlet_BCs_flt_gpu(sp);
#endif
#if defined(BC_T_SF_ROTATED)
    sf_open_BCs_flt_gpu(sp);
#endif
#ifdef BC_T_OPEN
    sf_open_v2_BCs_flt_gpu(sp);
#endif
}

void (*apply_BCs_on_spinor_field_flt)(spinor_field_flt *sp) = apply_BCs_on_spinor_field_flt_gpu;
#endif

void apply_BCs_on_spinor_field_flt_cpu(spinor_field_flt *sp) {
#if defined(BC_T_SF)
    sf_Dirichlet_BCs_flt_cpu(sp);
#endif
#if defined(BC_T_SF_ROTATED)
    sf_open_BCs_flt_cpu(sp);
#endif
#ifdef BC_T_OPEN
    sf_open_v2_BCs_flt_cpu(sp);
#endif
}

#ifndef WITH_GPU
void (*apply_BCs_on_spinor_field_flt)(spinor_field_flt *sp) = apply_BCs_on_spinor_field_flt_cpu;
#endif

#ifdef WITH_GPU
void apply_BCs_on_clover_term_gpu(clover_term *cl) {
#if (defined(WITH_EXPCLOVER) || defined(WITH_CLOVER)) && defined(BC_T_OPEN)
    cl_open_BCs_gpu(cl);
#endif

#if (defined(WITH_EXPCLOVER) || defined(WITH_CLOVER)) && (defined(BC_T_SF) || defined(BC_T_SF_ROTATED))
    cl_SF_BCs_gpu(cl);
#endif
}

void (*apply_BCs_on_clover_term)(clover_term *cl) = apply_BCs_on_clover_term_gpu;
#endif

void apply_BCs_on_clover_term_cpu(clover_term *cl) {
#if (defined(WITH_EXPCLOVER) || defined(WITH_CLOVER)) && defined(BC_T_OPEN)
    cl_open_BCs_cpu(cl);
#endif

#if (defined(WITH_EXPCLOVER) || defined(WITH_CLOVER)) && (defined(BC_T_SF) || defined(BC_T_SF_ROTATED))
    cl_SF_BCs_cpu(cl);
#endif
}

#ifndef WITH_GPU
void (*apply_BCs_on_clover_term)(clover_term *cl) = apply_BCs_on_clover_term_cpu;
#endif
