/*******************************************************************************
 *
 * Computation of the mesonic spectrum
 *
 *******************************************************************************/

#include "libhr.h"
#include <string.h>

#if defined(BC_T_SF_ROTATED) && defined(BC_T_SF)
#error The implementation of the Schroedinger functional has not been tested on this code
#endif

#ifdef FERMION_THETA
#error This code does not work with the fermion twisting !!!
#endif

/* Mesons parameters */
typedef struct input_ata_qprop {
    char mstring[256];
    ata_qprop_pars pars;

    /* for the reading function */
    input_record_t read[14];

} input_ata_qprop;

#define init_input_ata_qprop(varname)                                                                                     \
    {                                                                                                                     \
        .read = {                                                                                                         \
            { "quark quenched masses", "masses = %s", STRING_T, (varname).mstring },                                      \
            { "number of eigenvalues", "n_eigenvalues = %d", INT_T, &((varname).pars.n_eigenvalues) },                    \
            { "eva nevt parameter", "eva_nevt = %d", INT_T, &((varname).pars.eva_nevt) },                                 \
            { "eva omega1 parameter", "eva_omega1 = %lf", DOUBLE_T, &((varname).pars.eva_omega1) },                       \
            { "eva omega2 parameter", "eva_omega2 = %lf", DOUBLE_T, &((varname).pars.eva_omega2) },                       \
            { "eva imax parameter", "eva_imax = %d", INT_T, &((varname).pars.eva_imax) },                                 \
            { "eva kmax parameter", "eva_kmax = %d", INT_T, &((varname).pars.eva_kmax) },                                 \
            { "order of the hopping expansion", "hopping_order = %d", INT_T, &((varname).pars.hopping_order) },           \
            { "number of steps for truncation", "n_truncation_steps = %d", INT_T, &((varname).pars.n_truncation_steps) }, \
            { "number of sources for truncation", "n_sources_truncation = %d", INT_T,                                     \
              &((varname).pars.n_sources_truncation) },                                                                   \
            { "number of sources for correction", "n_sources_correction = %d", INT_T,                                     \
              &((varname).pars.n_sources_correction) },                                                                   \
            { "inverter precision", "inverter_precision = %lf", DOUBLE_T, &((varname).pars.inverter_precision) },         \
            { "dilution mode", "dilution = %d", INT_T, &((varname).pars.dilution) },                                      \
            { NULL, NULL, INT_T, NULL }                                                                                   \
        }                                                                                                                 \
    }

char cnfg_filename[256] = "";
char list_filename[256] = "";
char input_filename[256] = "input_file";
char output_filename[256] = "hairpins.out";
enum { UNKNOWN_CNFG, DYNAMICAL_CNFG, QUENCHED_CNFG };

input_ata_qprop ata_qprop_var = init_input_ata_qprop(ata_qprop_var);

typedef struct {
    char string[256];
    int t, x, y, z;
    int nc, nf;
    double b, m;
    int n;
    int type;
} filename_t;

int parse_cnfg_filename(char *filename, filename_t *fn) {
    int hm;
    char *tmp = NULL;
    char *basename;

    basename = filename;
    while ((tmp = strchr(basename, '/')) != NULL) {
        basename = tmp + 1;
    }

#ifdef REPR_FUNDAMENTAL
#define repr_name "FUN"
#elif defined REPR_SYMMETRIC
#define repr_name "SYM"
#elif defined REPR_ANTISYMMETRIC
#define repr_name "ASY"
#elif defined REPR_ADJOINT
#define repr_name "ADJ"
#endif
    hm = sscanf(basename, "%*[^_]_%dx%dx%dx%d%*[Nn]c%dr" repr_name "%*[Nn]f%db%lfm%lfn%d", &(fn->t), &(fn->x), &(fn->y),
                &(fn->z), &(fn->nc), &(fn->nf), &(fn->b), &(fn->m), &(fn->n));
    if (hm == 9) {
        fn->m = -fn->m; /* invert sign of mass */
        fn->type = DYNAMICAL_CNFG;
        return DYNAMICAL_CNFG;
    }
#undef repr_name

    double kappa;
    hm = sscanf(basename, "%dx%dx%dx%d%*[Nn]c%d%*[Nn]f%db%lfk%lfn%d", &(fn->t), &(fn->x), &(fn->y), &(fn->z), &(fn->nc),
                &(fn->nf), &(fn->b), &kappa, &(fn->n));
    if (hm == 9) {
        fn->m = .5 / kappa - 4.;
        fn->type = DYNAMICAL_CNFG;
        return DYNAMICAL_CNFG;
    }

    hm = sscanf(basename, "%*[^_]_%dx%dx%dx%d%*[Nn]c%db%lfn%d", &(fn->t), &(fn->x), &(fn->y), &(fn->z), &(fn->nc), &(fn->b),
                &(fn->n));
    if (hm == 7) {
        fn->type = QUENCHED_CNFG;
        return QUENCHED_CNFG;
    }

    hm = sscanf(basename, "%dx%dx%dx%d%*[Nn]c%db%lfn%d", &(fn->t), &(fn->x), &(fn->y), &(fn->z), &(fn->nc), &(fn->b), &(fn->n));
    if (hm == 7) {
        fn->type = QUENCHED_CNFG;
        return QUENCHED_CNFG;
    }

    fn->type = UNKNOWN_CNFG;
    return UNKNOWN_CNFG;
}

void read_cmdline(int argc, char *argv[]) {
    int i, ai = 0, ao = 0, ac = 0, al = 0, am = 0;
    FILE *list = NULL;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-i") == 0) {
            ai = i + 1;
        } else if (strcmp(argv[i], "-o") == 0) {
            ao = i + 1;
        } else if (strcmp(argv[i], "-c") == 0) {
            ac = i + 1;
        } else if (strcmp(argv[i], "-l") == 0) {
            al = i + 1;
        } else if (strcmp(argv[i], "-m") == 0) {
            am = i;
        }
    }

    if (am != 0) {
        print_compiling_info();
        exit(0);
    }

    if (ao != 0) { strcpy(output_filename, argv[ao]); }
    if (ai != 0) { strcpy(input_filename, argv[ai]); }

    error((ac == 0 && al == 0) || (ac != 0 && al != 0), 1, "parse_cmdline [mk_hairpins.c]",
          "Syntax: mk_hairpins { -c <config file> | -l <list file> } [-i <input file>] [-o <output file>] [-m]");

    if (ac != 0) {
        strcpy(cnfg_filename, argv[ac]);
        strcpy(list_filename, "");
    } else if (al != 0) {
        strcpy(list_filename, argv[al]);
        error((list = fopen(list_filename, "r")) == NULL, 1, "parse_cmdline [mk_hairpins.c]", "Failed to open list file\n");
        error(fscanf(list, "%s", cnfg_filename) == 0, 1, "parse_cmdline [mk_hairpins.c]", "Empty list file\n");
        fclose(list);
    }
}

int main(int argc, char *argv[]) {
    int i, k, n, dt;
    char tmp[256], *cptr;
    FILE *list;
    complex **ata_qprop[2];
    double *haircorr;
    complex *ctmp[3];
    filename_t fpars;
    double disc;

    /* setup process id and communications */
    setup_process(&argc, &argv);
    setup_gauge_fields();

    read_input(ata_qprop_var.read, input_filename);
    ata_qprop_var.pars.n_masses = 0;
    if (fpars.type == DYNAMICAL_CNFG) {
        ata_qprop_var.pars.n_masses = 1;
        ata_qprop_var.pars.mass[0] = fpars.m;
    } else if (fpars.type == QUENCHED_CNFG) {
        strcpy(tmp, ata_qprop_var.mstring);
        cptr = strtok(tmp, ";");
        ata_qprop_var.pars.n_masses = 0;
        while (cptr != NULL) {
            ata_qprop_var.pars.mass[ata_qprop_var.pars.n_masses] = atof(cptr);
            ata_qprop_var.pars.n_masses++;
            cptr = strtok(NULL, ";");
        }
    }

    /* Print inversion/truncation parameters */
    lprintf("MAIN", 0, "Simulation masses:");
    for (i = 0; i < ata_qprop_var.pars.n_masses; i++) {
        lprintf("MAIN", 0, " %e", ata_qprop_var.pars.mass[i]);
    }
    lprintf("MAIN", 0, "\n");
    lprintf("MAIN", 0, "Eigenvalues parameters (nev, nevt, omega1, omega2, kmax, imax): %d, %d, %e, %e, %d, %d\n",
            ata_qprop_var.pars.n_eigenvalues, ata_qprop_var.pars.eva_nevt, ata_qprop_var.pars.eva_omega1,
            ata_qprop_var.pars.eva_omega2, ata_qprop_var.pars.eva_kmax, ata_qprop_var.pars.eva_imax);
    lprintf("MAIN", 0, "Hopping parameter expansion order (Disabled = -1): %d\n", ata_qprop_var.pars.hopping_order);
    lprintf("MAIN", 0, "Truncate after # steps: %d\n", ata_qprop_var.pars.n_truncation_steps);
    lprintf("MAIN", 0, "Number of truncation sources: %d\n", ata_qprop_var.pars.n_sources_truncation);
    lprintf("MAIN", 0, "Number of correction sources: %d\n", ata_qprop_var.pars.n_sources_correction);
    lprintf("MAIN", 0, "Inverter precision: %e\n", ata_qprop_var.pars.inverter_precision);
    lprintf("MAIN", 0, "Dilution: %d\n", ata_qprop_var.pars.dilution);

    ata_qprop_init(&(ata_qprop_var.pars));

    haircorr = (double *)malloc(GLB_T * sizeof(double));

    ata_qprop[0] = (complex **)malloc(sizeof(complex *) * ata_qprop_var.pars.n_masses);
    ata_qprop[1] = (complex **)malloc(sizeof(complex *) * ata_qprop_var.pars.n_masses);
    for (i = 0; i < ata_qprop_var.pars.n_masses; ++i) {
        ata_qprop[0][i] = (complex *)malloc(sizeof(complex) * 16 * GLB_T);
        ata_qprop[1][i] = (complex *)malloc(sizeof(complex) * 16 * GLB_T);
    }

    ctmp[0] = (complex *)malloc(GLB_T * sizeof(complex));
    ctmp[1] = (complex *)malloc(GLB_T * sizeof(complex));
    ctmp[2] = (complex *)malloc(GLB_T * sizeof(complex));

    list = NULL;
    if (strcmp(list_filename, "") != 0) {
        error((list = fopen(list_filename, "r")) == NULL, 1, "main [mk_hairpins.c]", "Failed to open list file\n");
    }

    i = 0;

    unsigned long start_time = clock();

    while (1) {
        if (list != NULL) {
            if (fscanf(list, "%s", cnfg_filename) == 0 || feof(list)) { break; }
        }

        i++;

        lprintf("MAIN", 0, "Configuration from %s\n", cnfg_filename);
        /* NESSUN CHECK SULLA CONSISTENZA CON I PARAMETRI DEFINITI !!! */
        read_gauge_field(cnfg_filename);
        represent_gauge_field();

        lprintf("TEST", 0, "<p> %1.6f\n", avr_plaquette());

        full_plaquette();

        traced_ata_qprop(ata_qprop, 2);

        for (k = 0; k < ata_qprop_var.pars.n_masses; ++k) {
            lprintf("MAIN", 0, "conf #%d mass=%2.6f \n", i, ata_qprop_var.pars.mass[k]);

            /*******************************************************************************
       * disc = +/- tr Gamma D^{-1}(0,0)
       *
       * (we are not sure about the sign, but it doesn't matter when you square it)
       *
       * haircorr[t] =
       *       1
       *      ---  sum    tr [GammaBar D^{-1}(0,0)] tr [Gamma D^{-1}(t,x,y,z;t,x,y,z)]
       *       V3  x,y,z
       *
       * GammaBar = g0 Gamma^dag g0
       *
       *******************************************************************************/

#define HAIRPIN(name)                                                                                           \
    for (n = 0; n < GLB_T; ++n) {                                                                               \
        name##_trace_H(ctmp[0] + n, ata_qprop[0][k] + 16 * n);                                                  \
        name##_trace_H(ctmp[1] + n, ata_qprop[1][k] + 16 * n);                                                  \
    }                                                                                                           \
    disc = 0.;                                                                                                  \
    for (dt = 0; dt < GLB_T; ++dt) {                                                                            \
        disc += (ctmp[0][dt].re + ctmp[1][dt].re) / (2. * GLB_T);                                               \
        haircorr[dt] = 0.;                                                                                      \
        for (n = 0; n < GLB_T; ++n)                                                                             \
            haircorr[dt] += ctmp[0][n].re * ctmp[1][safe_mod(n + dt, GLB_T)].re / GLB_T;                        \
    }                                                                                                           \
    for (dt = 0; dt < GLB_T; ++dt)                                                                              \
        haircorr[dt] *= _BAR_SIGN_;                                                                             \
    lprintf("MAIN", 0, "\n");                                                                                   \
    lprintf("MAIN", 0, "conf #%d mass=%2.6f HAIRPIN " #name "= ", i, ata_qprop_var.pars.mass[k]);               \
    for (n = 0; n < GLB_T; ++n) {                                                                               \
        lprintf("MAIN", 0, "%e ", haircorr[n]);                                                                 \
    }                                                                                                           \
    lprintf("MAIN", 0, "\n");                                                                                   \
    lprintf("MAIN", 0, "conf #%d mass=%2.6f SINGLETRACE " #name "= %e\n", i, ata_qprop_var.pars.mass[k], disc); \
    fflush(stdout)

            /*      disc=0.;\*/
            /*      for(dt=0;dt<GLB_T;++dt) {\*/
            /*              disc += ctmp[2][dt].re/GLB_T;\*/
            /*              haircorr[dt]=0.;\*/
            /*              for(n=0;n<GLB_T;++n)\*/
            /*                      haircorr[dt] += ctmp[2][n].re*ctmp[2][safe_mod(n+dt,GLB_T)].re/GLB_T;\*/
            /*      }\*/

#define _BAR_SIGN_ 1
            HAIRPIN(id);
#undef _BAR_SIGN_

#define _BAR_SIGN_ -1
            HAIRPIN(g5);
#undef _BAR_SIGN_

#define _BAR_SIGN_ 1
            HAIRPIN(g0);
#undef _BAR_SIGN_

#define _BAR_SIGN_ 1
            HAIRPIN(g0g5);
#undef _BAR_SIGN_

#define _BAR_SIGN_ -1
            HAIRPIN(g1);
#undef _BAR_SIGN_

#define _BAR_SIGN_ -1
            HAIRPIN(g2);
#undef _BAR_SIGN_

#define _BAR_SIGN_ -1
            HAIRPIN(g3);
#undef _BAR_SIGN_

#define _BAR_SIGN_ 1
            HAIRPIN(g0g1);
#undef _BAR_SIGN_

#define _BAR_SIGN_ 1
            HAIRPIN(g0g2);
#undef _BAR_SIGN_

#define _BAR_SIGN_ 1
            HAIRPIN(g0g3);
#undef _BAR_SIGN_

#define _BAR_SIGN_ -1
            HAIRPIN(g5g1);
#undef _BAR_SIGN_

#define _BAR_SIGN_ -1
            HAIRPIN(g5g2);
#undef _BAR_SIGN_

#define _BAR_SIGN_ -1
            HAIRPIN(g5g3);
#undef _BAR_SIGN_

#define _BAR_SIGN_ -1
            HAIRPIN(g0g5g1);
#undef _BAR_SIGN_

#define _BAR_SIGN_ -1
            HAIRPIN(g0g5g2);
#undef _BAR_SIGN_

#define _BAR_SIGN_ -1
            HAIRPIN(g0g5g3);
#undef _BAR_SIGN_
        }

        if (list == NULL) { break; }
    }

    if (list != NULL) { fclose(list); }

    ata_qprop_free();

    for (i = 0; i < ata_qprop_var.pars.n_masses; ++i) {
        free(ata_qprop[0][i]);
        free(ata_qprop[1][i]);
    }
    free(ata_qprop[0]);
    free(ata_qprop[1]);

    free(haircorr);

    free(ctmp[0]);
    free(ctmp[1]);
    free(ctmp[2]);

    free_suNg_field(u_gauge);
#ifndef REPR_FUNDAMENTAL
    free_suNf_field(u_gauge_f);
#endif

    lprintf("MAIN", 0, "Execution time    = %f hours\n", (clock() - start_time) * 1.0 / CLOCKS_PER_SEC / 3600);
    lprintf("MAIN", 0, "Average time/cnfg = %f hours\n", (clock() - start_time) * 1.0 / CLOCKS_PER_SEC / 3600 / i);

    return 0;
}
