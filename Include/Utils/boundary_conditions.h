#ifndef BOUNDARY_CONDITIONS_H
#define BOUNDARY_CONDITIONS_H

#include "suN_types.h"
#include "spinor_field.h"

#ifdef __cplusplus
extern "C" {
#endif

// boundary_conditions.c
typedef struct {
    double gauge_boundary_improvement_cs;
    double gauge_boundary_improvement_ct;
    double chiSF_boundary_improvement_ds;
    double fermion_twisting_theta[4];
    int SF_BCs;
    suNg gauge_boundary_up;
    suNg gauge_boundary_dn;
} BCs_pars_t;

// Architecture independent functions
#if (defined(WITH_EXPCLOVER) || defined(WITH_CLOVER)) && (defined(BC_T_SF) || defined(BC_T_SF_ROTATED))
extern void (*cl_SF_BCs)(clover_term *cl);
#endif

#if (defined(WITH_EXPCLOVER) || defined(WITH_CLOVER)) && defined(BC_T_OPEN)
extern void (*cl_open_BCs)(clover_term *cl);
#endif

#ifdef BC_T_ANTIPERIODIC
extern void (*sp_T_antiperiodic_BCs)();
#endif

#ifdef BC_X_ANTIPERIODIC
extern void (*sp_X_antiperiodic_BCs)();
#endif

#ifdef BC_Y_ANTIPERIODIC
extern void (*sp_Y_antiperiodic_BCs)();
#endif

#ifdef BC_Z_ANTIPERIODIC
extern void (*sp_Z_antiperiodic_BCs)();
#endif

#ifdef BC_T_SF_ROTATED
extern void (*chiSF_ds_BT)(double ds);
#endif

#if defined(BC_T_SF) || defined(BC_T_SF_ROTATED)
extern void (*gf_SF_BCs)(suNg *dn, suNg *up);
extern void (*SF_classical_solution_core)(suNg *U, int it);
void SF_classical_solution(void);
#endif

#ifdef BC_T_OPEN
extern void (*gf_open_BCs)();
#endif

#if defined(BC_T_SF) || defined(BC_T_SF_ROTATED)
extern void (*mf_Dirichlet_BCs)(suNg_av_field *force);
#endif

#ifdef BC_T_OPEN
extern void (*mf_open_BCs)(suNg_av_field *force);
#endif

#if defined(BC_T_SF)
extern void (*sf_Dirichlet_BCs)(spinor_field *sp);
extern void (*sf_Dirichlet_BCs_flt)(spinor_field_flt *sp);
#endif

#if defined(BC_T_SF_ROTATED)
extern void (*sf_open_BCs)(spinor_field *sp);
extern void (*sf_open_BCs_flt)(spinor_field_flt *sp);
#endif

#ifdef BC_T_OPEN
extern void (*sf_open_v2_BCs)(spinor_field *sf);
extern void (*sf_open_v2_BCs_flt)(spinor_field_flt *sp);
#endif

#ifdef WITH_GPU
#ifdef __cplusplus
extern "C" {
#endif
// For testing, core functions need to expose the architecture
#if (defined(WITH_EXPCLOVER) || defined(WITH_CLOVER)) && (defined(BC_T_SF) || defined(BC_T_SF_ROTATED))
void cl_SF_BCs_gpu(clover_term *cl);
#endif

#if (defined(WITH_EXPCLOVER) || defined(WITH_CLOVER)) && defined(BC_T_OPEN)
void cl_open_BCs_gpu(clover_term *cl);
#endif

#ifdef BC_T_ANTIPERIODIC
void sp_T_antiperiodic_BCs_gpu();
#endif

#ifdef BC_X_ANTIPERIODIC
void sp_X_antiperiodic_BCs_gpu();
#endif

#ifdef BC_Y_ANTIPERIODIC
void sp_Y_antiperiodic_BCs_gpu();
#endif

#ifdef BC_Z_ANTIPERIODIC
void sp_Z_antiperiodic_BCs_gpu();
#endif

#ifdef BC_T_SF_ROTATED
void chiSF_ds_BT_gpu(double ds);
#endif

#if defined(BC_T_SF) || defined(BC_T_SF_ROTATED)
void gf_SF_BCs_gpu(suNg *dn_h, suNg *up_h);
void SF_classical_solution_core_gpu(suNg *U_h, int it);
#endif

#ifdef BC_T_OPEN
void gf_open_BCs_gpu();
#endif

#if defined(BC_T_SF) || defined(BC_T_SF_ROTATED)
void mf_Dirichlet_BCs_gpu(suNg_av_field *force);
#endif

#ifdef BC_T_OPEN
void mf_open_BCs_gpu(suNg_av_field *force);
#endif

#if defined(BC_T_SF)
void sf_Dirichlet_BCs_gpu(spinor_field *sp);
void sf_Dirichlet_BCs_flt_gpu(spinor_field_flt *sp);
#endif

#if defined(BC_T_SF_ROTATED)
void sf_open_BCs_gpu(spinor_field *sp);
void sf_open_BCs_flt_gpu(spinor_field_flt *sp);
#endif

#ifdef BC_T_OPEN
void sf_open_v2_BCs_gpu(spinor_field *sp);
void sf_open_v2_BCs_flt_gpu(spinor_field_flt *sp);
#endif
#ifdef __cplusplus
}
#endif
#endif

// For testing, core functions need to expose the architecture
#if (defined(WITH_EXPCLOVER) || defined(WITH_CLOVER)) && (defined(BC_T_SF) || defined(BC_T_SF_ROTATED))
void cl_SF_BCs_cpu(clover_term *cl);
#endif

#if (defined(WITH_EXPCLOVER) || defined(WITH_CLOVER)) && defined(BC_T_OPEN)
void cl_open_BCs_cpu(clover_term *cl);
#endif

#ifdef BC_T_ANTIPERIODIC
void sp_T_antiperiodic_BCs_cpu();
#endif

#ifdef BC_X_ANTIPERIODIC
void sp_X_antiperiodic_BCs_cpu();
#endif

#ifdef BC_Y_ANTIPERIODIC
void sp_Y_antiperiodic_BCs_cpu();
#endif

#ifdef BC_Z_ANTIPERIODIC
void sp_Z_antiperiodic_BCs_cpu();
#endif

#ifdef BC_T_SF_ROTATED
void chiSF_ds_BT_cpu(double ds);
#endif

#if defined(BC_T_SF) || defined(BC_T_SF_ROTATED)
void gf_SF_BCs_cpu(suNg *dn, suNg *up);
void SF_classical_solution_core_cpu(suNg *U, int it);
#endif

#ifdef BC_T_OPEN
void gf_open_BCs_cpu();
#endif

#if defined(BC_T_SF) || defined(BC_T_SF_ROTATED)
void mf_Dirichlet_BCs_cpu(suNg_av_field *force);
#endif

#ifdef BC_T_OPEN
void mf_open_BCs_cpu(suNg_av_field *force);
#endif

#if defined(BC_T_SF)
void sf_Dirichlet_BCs_cpu(spinor_field *sp);
void sf_Dirichlet_BCs_flt_cpu(spinor_field_flt *sp);
#endif

#if defined(BC_T_SF_ROTATED)
void sf_open_BCs_cpu(spinor_field *sp);
void sf_open_BCs_flt_cpu(spinor_field_flt *sp);
#endif

#ifdef BC_T_OPEN
void sf_open_v2_BCs_cpu(spinor_field *sf);
void sf_open_v2_BCs_flt_cpu(spinor_field_flt *sp);
#endif

#ifdef GAUGE_SPATIAL_TWIST
void init_plaq_twisted_BCs();
#endif

void init_BCs(BCs_pars_t *pars);
void init_plaq_open_BCs(double *plaq_weight, double *rect_weight, double ct, double cs);
void free_BCs();

extern void (*apply_BCs_on_represented_gauge_field)();
void apply_BCs_on_represented_gauge_field_cpu();
#ifdef WITH_GPU
void apply_BCs_on_represented_gauge_field_gpu();
#endif

extern void (*apply_BCs_on_fundamental_gauge_field)();
void apply_BCs_on_fundamental_gauge_field_cpu();
#ifdef WITH_GPU
void apply_BCs_on_fundamental_gauge_field_gpu();
#endif

extern void (*apply_BCs_on_momentum_field)(suNg_av_field *force);
void apply_BCs_on_momentum_field_cpu(suNg_av_field *force);
#ifdef WITH_GPU
void apply_BCs_on_momentum_field_gpu(suNg_av_field *force);
#endif

extern void (*apply_BCs_on_spinor_field)(spinor_field *sp);
void apply_BCs_on_spinor_field_cpu(spinor_field *sp);
#ifdef WITH_GPU
void apply_BCs_on_spinor_field_gpu(spinor_field *sp);
#endif

extern void (*apply_BCs_on_spinor_field_flt)(spinor_field_flt *sp);
void apply_BCs_on_spinor_field_flt_cpu(spinor_field_flt *sp);
#ifdef WITH_GPU
void apply_BCs_on_spinor_field_flt_gpu(spinor_field_flt *sp);
#endif

extern void (*apply_BCs_on_clover_term)(clover_term *cl);
void apply_BCs_on_clover_term_cpu(clover_term *cl);
#ifdef WITH_GPU
void apply_BCs_on_clover_term_gpu(clover_term *cl);
#endif

void apply_background_field_zdir(suNg_field *V, double Q, int n);

// Schrodinger functional helpers
void calc_SF_U(suNg *U, int x0);
void init_gf_SF_BCs(suNg *dn, suNg *up);
void init_plaq_SF_BCs(double ct);

#ifdef __cplusplus
}
#endif
#endif //BOUNDARY_CONDITIONS_H
