/*******************************************************************************
 *
 *
 *******************************************************************************/

#include "libhr.h"

rhmc_par _update_par;

int main(int argc, char *argv[]) {
    setup_process(&argc, &argv);
    char tmp[256];

    BCs_pars_t BCs_pars = { .fermion_twisting_theta = { 0., M_PI / 5., M_PI / 5., M_PI / 5. },

                            /*      .fermion_twisting_theta={0.,0.,0.,0.},  */

                            .gauge_boundary_improvement_cs = 1.,
                            .gauge_boundary_improvement_ct = 1.,
                            .chiSF_boundary_improvement_ds = 0.5,
                            .SF_BCs = 0 };

    double acc = 1.e-20;

    double mass = -0.3099358667;
    /*  double mass=0.0; */

    _update_par.mass = -0.3099358667;
    _update_par.SF_ds = 0.5;
    _update_par.SF_sign = 1;
    _update_par.SF_ct = 1;
    _update_par.SF_zf = 1.3;

    logger_setlevel(0, 100); /* log all */
    if (PID != 0) {
        logger_disable();
    } else {
        sprintf(tmp, ">out_%d", PID);
        logger_stdout(tmp);
        sprintf(tmp, "err_%d", PID);
        freopen(tmp, "w", stderr);
    }

    lprintf("MAIN", 0, "PId =  %d [world_size: %d]\n\n", PID, WORLD_SIZE);

    read_input(glb_var.read, "test_input");
    read_input(rlx_var.read, "test_input");

    rlxd_init(rlx_var.rlxd_level, rlx_var.rlxd_seed);

#if NG != 3 || NF != 3
#error "Can work only with NC=3 and Nf==3"
#endif

    /* setup communication geometry */
    if (geometry_init() == 1) {
        finalize_process();
        return 0;
    }

    geometry_mpi_eo();

    init_BCs(&BCs_pars);

    lprintf("MAIN", 0, "This test implements a comparison with a working code of Stefan Sint\n");

    u_gauge = alloc_suNg_field(&glattice);
    u_gauge_f = alloc_suNf_field(&glattice);

    /* unit_u(u_gauge); */
    read_gauge_field_nocheck("suNg_field_sint.dat");
    apply_BCs_on_fundamental_gauge_field();

    lprintf("MAIN", 0, "mass = %f\n", mass);
    /*   lprintf("MAIN",0,"ds = %f\n",_update_par.SF_ds); */
    /*   lprintf("MAIN",0,"zf = %f\n",_update_par.SF_zf); */
    /*   lprintf("MAIN",0,"theta = %f\n",_update_par.SF_theta); */
    /*   lprintf("MAIN",0,"sign = %d\n",_update_par.SF_sign); */
    lprintf("MAIN", 0, "To be compared with results in file gxly_testconfig\n");

    represent_gauge_field();
    full_plaquette(u_gauge);

    SF_PCAC_wall_corr(mass, acc, NULL);

    free_suNg_field(u_gauge);
    free_suNf_field(u_gauge_f);

    free_BCs();
    finalize_process();
    exit(0);
}
