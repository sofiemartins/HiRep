/*******************************************************************************
*
* Check of GPU boundary conditions (antiperiodic)
* NOCOMPILE= !WITH_GPU
*
*******************************************************************************/

#include "libhr.h"

int main(int argc, char *argv[]) {
    int errors = 0;
    setup_process(&argc, &argv);
    setup_gauge_fields();

    lprintf("MAIN", 0, "Generating a random gauge field... ");
    random_u(u_gauge);
    start_sendrecv_suNg_field(u_gauge);
    represent_gauge_field();
    lprintf("MAIN", 0, "done.\n\n");

    suNf_field *tmp = alloc(tmp, 1, &glattice);

    copy_from_gpu_suNf_field(u_gauge_f);

    lprintf("INFO", 0, "Testing boundary conditions for represented gauge fields\n");

#ifdef BC_T_ANTIPERIODIC
    sp_T_antiperiodic_BCs_cpu();
    sp_T_antiperiodic_BCs_gpu();
#endif

#ifdef BC_X_ANTIPERIODIC
    sp_X_antiperiodic_BCs_cpu();
    sp_X_antiperiodic_BCs_gpu();
#endif

#ifdef BC_Y_ANTIPERIODIC
    sp_Y_antiperiodic_BCs_cpu();
    sp_Y_antiperiodic_BCs_gpu();
#endif

#ifdef BC_Z_ANTIPERIODIC
    sp_Z_antiperiodic_BCs_cpu();
    sp_Z_antiperiodic_BCs_gpu();
#endif

#ifdef BC_T_SF_ROTATED // Note: does not compile at the moment, but it does not harm to have it here anyway
    double ds = 0.1;
    chiSF_ds_BT_cpu(ds);
    chiSF_ds_BT_gpu(ds);
#endif

    copy_cpu(tmp, u_gauge_f);
    copy_to_gpu_suNf_field(tmp);
    sub_assign(tmp, u_gauge_f);
    double diffnorm = max(tmp);
    errors = check_diff_norm(diffnorm, 1e-14);

    lprintf("INFO", 0, "Testing boundary conditions for fundamental gauge field\n");
    suNg_field *tmp_fund = alloc(tmp_fund, 1, &glattice);
    copy_from_gpu_suNg_field(u_gauge);

#if defined(BC_T_SF) || defined(BC_T_SF_ROTATED)
    BCs_pars_t pars;
    init_gf_SF_BCs(&(pars.gauge_boundary_dn), &(pars.gauge_boundary_up));
    gf_SF_BCs_cpu(&(pars.gauge_boundary_dn), &(pars.gauge_boundary_up));
    gf_SF_BCs_gpu(&(pars.gauge_boundary_dn), &(pars.gauge_boundary_up));
#endif

#ifdef BC_T_OPEN
    gf_open_BCs_cpu();
    gf_open_BCs_gpu();
#endif

    copy_cpu(tmp_fund, u_gauge);
    copy_to_gpu(tmp_fund);
    sub_assign(tmp_fund, u_gauge);
    diffnorm = max(tmp_fund);
    errors += check_diff_norm(diffnorm, 1e-14);

    lprintf("INFO", 0, "Testing boundary conditions for momentum fields\n");

    suNg_av_field *force = alloc(force, 2, &glattice);
    suNg_av_field *tmp_force = force + 1;
    random_suNg_av_field_cpu(force);
    copy_to_gpu(force);

#if defined(BC_T_SF) || defined(BC_T_SF_ROTATED)
    mf_Dirichlet_BCs_cpu(force);
    mf_Dirichlet_BCs_gpu(force);
#endif

#ifdef BC_T_OPEN
    mf_open_BCs_cpu(force);
    mf_open_BCs_gpu(force);
#endif

    copy_cpu(tmp_force, force);
    copy_to_gpu_suNg_av_field(tmp_force);
    sub_assign(tmp_force, force);
    diffnorm = max(tmp_force);
    errors += check_diff_norm(diffnorm, 1e-14);

    lprintf("INFO", 0, "Testing boundary conditions for double precision spinor fields\n");

    spinor_field *sf = alloc(sf, 2, &glat_odd);
    spinor_field *sf_tmp = sf + 1;
    gaussian_spinor_field(sf);
    copy_from_gpu(sf);

#if defined(BC_T_SF)
    sf_Dirichlet_BCs_cpu(sf);
    sf_Dirichlet_BCs_gpu(sf);
#endif

#if defined(BC_T_SF_ROTATED)
    sf_open_BCs_cpu(sf);
    sf_open_BCs_gpu(sf);
#endif

#ifdef BC_T_OPEN
    sf_open_v2_BCs_cpu(sf);
    sf_open_v2_BCs_gpu(sf);
#endif

    copy_cpu(sf_tmp, sf);
    copy_to_gpu(sf_tmp);
    sub_assign(sf_tmp, sf);
    diffnorm = max(sf_tmp);
    errors += check_diff_norm(diffnorm, 1e-14);

    lprintf("INFO", 0, "Testing boundary conditions for single precision spinor fields\n");

    spinor_field_flt *sfflt = alloc(sfflt, 2, &glat_even);
    spinor_field_flt *sfflt_tmp = sfflt + 1;
    gaussian_spinor_field_flt(sfflt);
    copy_from_gpu(sfflt);

#if defined(BC_T_SF)
    sf_Dirichlet_BCs_flt_cpu(sfflt);
    sf_Dirichlet_BCs_flt_gpu(sfflt);
#endif

#if defined(BC_T_SF_ROTATED)
    sf_open_BCs_flt_cpu(sfflt);
    sf_open_BCs_flt_gpu(sfflt);
#endif

#ifdef BC_T_OPEN
    sf_open_v2_BCs_flt_cpu(sfflt);
    sf_open_v2_BCs_flt_gpu(sfflt);
#endif

    copy_cpu(sfflt_tmp, sfflt);
    copy_to_gpu(sfflt_tmp);
    sub_assign(sfflt_tmp, sfflt);
    diffnorm = max(sfflt_tmp);
    errors += check_diff_norm(diffnorm, 1e-14);

    lprintf("INFO", 0, "Testing boundary conditions for clover terms\n");

    clover_term *cl = alloc(cl, 2, &glattice);
    clover_term *cl_tmp = cl + 1;
    random_clover_term_cpu(cl);
    copy_to_gpu(cl);

#if (defined(WITH_EXPCLOVER) || defined(WITH_CLOVER)) && (defined(BC_T_SF) || defined(BC_T_SF_ROTATED))
    cl_SF_BCs_cpu(cl);
    cl_SF_BCs_gpu(cl);
#endif

#if (defined(WITH_EXPCLOVER) || defined(WITH_CLOVER)) && defined(BC_T_OPEN)
    cl_open_BCs_cpu(cl);
    cl_open_BCs_gpu(cl);
#endif

    copy_cpu(cl_tmp, cl);
    copy_to_gpu(cl_tmp);

    // printf("sqnorm: %0.15e\n", max(cl_tmp));
    //lprintf("SANITY", 0, "val1: %0.15e, val2: %0.15e\n", sqnorm(cl_tmp), sqnorm(cl));
    sub_assign(cl_tmp, cl);
    diffnorm = max(cl_tmp);
    errors += check_diff_norm(diffnorm, 1e-14);

    finalize_process();
    return errors;
}
