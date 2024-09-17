/*******************************************************************************
 *
 * Gauge covariance of the SF observables
 *
 *******************************************************************************/

#include "libhr.h"

rhmc_par _update_par = { 0 };
static suNg_field *g;

static void random_g(void) {
    _MASTER_FOR(&glattice, ix) {
        random_suNg(_FIELD_AT(g, ix));
    }

    if (COORD[0] == 0) {
        for (int ix1 = 0; ix1 < X; ++ix1) {
            for (int iy1 = 0; iy1 < Y; ++iy1) {
                for (int iz1 = 0; iz1 < Z; ++iz1) {
                    /*      ix=ipt(2,ix1,iy1,iz1);
                _suNg_unit(*_FIELD_AT(g,ix));*/
                    int ix = ipt(1, ix1, iy1, iz1);
                    _suNg_unit(*_FIELD_AT(g, ix));
                    ix = ipt(0, ix1, iy1, iz1);
                    _suNg_unit(*_FIELD_AT(g, ix));
                }
            }
        }
    }

    if (COORD[0] == NP_T - 1) {
        for (int ix1 = 0; ix1 < X; ++ix1) {
            for (int iy1 = 0; iy1 < Y; ++iy1) {
                for (int iz1 = 0; iz1 < Z; ++iz1) {
                    int ix = ipt(T - 1, ix1, iy1, iz1);
                    _suNg_unit(*_FIELD_AT(g, ix));
                    /*      ix=ipt(T-2,ix1,iy1,iz1);
                _suNg_unit(*_FIELD_AT(g,ix));
                ix=ipt(T-3,ix1,iy1,iz1);
                _suNg_unit(*_FIELD_AT(g,ix));*/
                }
            }
        }
    }
}

static void transform_u(void) {
    _MASTER_FOR(&glattice, ix) {
        for (int mu = 0; mu < 4; mu++) {
            int iy = iup(ix, mu);
            suNg *u = pu_gauge(ix, mu);
            suNg v;

            _suNg_times_suNg_dagger(v, *u, *_FIELD_AT(g, iy));
            _suNg_times_suNg(*u, *_FIELD_AT(g, ix), v);
        }
    }

    start_sendrecv_suNg_field(u_gauge);
    represent_gauge_field();
}

int main(int argc, char *argv[]) {
    setup_process(&argc, &argv);
    setup_gauge_fields();

    double acc = 1.e-20;

    BCs_pars_t BCs_pars = { .fermion_twisting_theta = { 0., M_PI / 5., M_PI / 5., M_PI / 5. },
                            .gauge_boundary_improvement_cs = 1.,
                            .gauge_boundary_improvement_ct = 1.,
                            .chiSF_boundary_improvement_ds = 0.5,
                            .SF_BCs = 0 };

    init_BCs(&BCs_pars);

    double mass = 0.0;
    _update_par.SF_zf = 6.;
    _update_par.SF_ds = 3.;
    _update_par.SF_sign = 1;

    random_u(u_gauge);
    apply_BCs_on_fundamental_gauge_field();

    represent_gauge_field();

    g = alloc_gtransf(&glattice);
    random_g();
    start_sendrecv_gtransf(g);
    complete_sendrecv_gtransf(g);

    lprintf("MAIN", 0, "Plaquette before the random gauge transf %f\n", avr_plaquette());
    SF_PCAC_wall_corr(mass, acc, NULL);

    transform_u();

    lprintf("MAIN", 0, "Plaquette after the random gauge transf %f\n", avr_plaquette());
    SF_PCAC_wall_corr(mass, acc, NULL);

    free_suNg_field(u_gauge);
    free_suNf_field(u_gauge_f);

    free_gtransf(g);

    finalize_process();
    exit(0);
}
