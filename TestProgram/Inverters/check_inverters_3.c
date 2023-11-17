/******************************************************************************
*
* Test of modules
*
******************************************************************************/

#include "libhr.h"

int nhb, nor, nit, nth, nms, level, seed;
double beta;

int main(int argc, char *argv[]) {
    int return_value = 0;

    int i;
    double tau;
    spinor_field *s1, *s2;
    spinor_field *res;

    mshift_par par;

    int cgiters;
    logger_map("DEBUG", "debug");

    setup_process(&argc, &argv);

    setup_gauge_fields();

    lprintf("MAIN", 0, "Generating a random gauge field... ");

    random_u(u_gauge);
    lprintf("MAIN", 0, "done.\n");

    start_sendrecv_suNg_field(u_gauge);
    complete_sendrecv_suNg_field(u_gauge);

    represent_gauge_field();

    par.n = 6;
    par.shift = (double *)malloc(sizeof(double) * (par.n));
    par.err2 = 1.e-28;
    par.max_iter = 0;
    res = alloc_spinor_field(par.n + 2,
#ifdef UPDATE_EO
                             &glat_even
#else
                             &glattice
#endif
    );
    s1 = res + par.n;
    s2 = s1 + 1;

    par.shift[0] = +0.1;
    par.shift[1] = -0.21;
    par.shift[2] = +0.05;
    par.shift[3] = -0.01;
    par.shift[4] = -0.15;
    par.shift[5] = -0.05;

    gaussian_spinor_field(s1);
    tau = spinor_field_sqnorm_f(s1);
    lprintf("QMR TEST", 0, "Initial Norm: %e\n", tau);

    /* TEST g5QMR_M */

    lprintf("QMR TEST", 0, "\n");
    lprintf("QMR TEST", 0, "Testing g5QMR multishift\n");
    lprintf("QMR TEST", 0, "------------------------\n");

    cgiters = g5QMR_mshift(&par, &D, s1, res);
    lprintf("QMR TEST", 0, "Converged in %d iterations\n", cgiters);

    for (i = 0; i < par.n; ++i) {
        D(s2, &res[i]);
        spinor_field_mul_add_assign_f(s2, -par.shift[i], &res[i]);
        spinor_field_sub_assign_f(s2, s1);
        tau = spinor_field_sqnorm_f(s2) / spinor_field_sqnorm_f(s1);
        lprintf("QMR TEST", 0, "test g5QMR[%d] = %e (req. %e)\n", i, tau, par.err2);
        if (tau > par.err2) { return_value += 1; }
    }

    free_spinor_field(res);
    free(par.shift);
    finalize_process();

    return return_value;
}
