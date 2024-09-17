/*******************************************************************************
 *
 *
 *******************************************************************************/

#include "libhr.h"

rhmc_par _update_par;

int main(int argc, char *argv[]) {
    setup_process(&argc, &argv);
    setup_gauge_fields();
    char sbuf[350];
    double a = 1.0;
    BCs_pars_t BCs_pars = { .fermion_twisting_theta = { 0., a * M_PI / 5., a * M_PI / 5., a * M_PI / 5. },

                            /*     .fermion_twisting_theta={0.,0.,0.,0.},    */

                            .gauge_boundary_improvement_cs = 1.,
                            .gauge_boundary_improvement_ct = 1.,
                            .chiSF_boundary_improvement_ds = 0.5,
                            .SF_BCs = 0 };

    double mass = 0.0;
    double acc = 1.e-20;

    _update_par.mass = 0.0;
    _update_par.SF_ds = 0.5;
    _update_par.SF_sign = 1;
    _update_par.SF_ct = 1.0;
    _update_par.SF_zf = 1.0;

    logger_map("DEBUG", "debug");

#if NG != 3 || NF != 3
#error "Can work only with NC=3 and Nf==3"
#endif

    init_BCs(&BCs_pars);

    lprintf("MAIN", 0, "This test implements a comparison with a working code of Stefan Sint\n");

    unit_u(u_gauge);

    apply_BCs_on_fundamental_gauge_field();

    lprintf("MAIN", 0, "mass = %f\n", mass);

    lprintf("MAIN", 0, "To be compared with results in file gxly_tree_b0_thpi5_new\n");

    represent_gauge_field();

    full_plaquette();

    SF_PCAC_wall_corr(mass, acc, NULL);

    free_suNg_field(u_gauge);
    free_suNf_field(u_gauge_f);

    free_BCs();
    finalize_process();
    exit(0);
}
