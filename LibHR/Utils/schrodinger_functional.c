/***************************************************************************\
* Copyright (c) 2008-2024, Agostino Patella, Claudio Pica, Sofie Martins    *
* All rights reserved.                                                      *
\***************************************************************************/

#include "libhr_core.h"
#include "io.h"
#include <math.h>
#include <stdlib.h>

#define ST 1.414213562373095048801688724209698078570

#if defined(BC_T_SF) || defined(BC_T_SF_ROTATED)
#ifndef GAUGE_SON

#if NG == 2

#ifndef HALFBG_SF
static double SF_eta = PI / 4.0;
static double SF_phi0_up[NG] = { -PI, PI };
#else
static double SF_eta = PI / 8.0;
static double SF_phi0_up[NG] = { -PI / 2, PI / 2 };
#endif

static double SF_phi0_dn[NG] = { 0., 0. };
static double SF_phi1_dn[NG] = { -1., 1. };
static double SF_phi1_up[NG] = { 1., -1. };

#elif NG == 3

static double SF_eta = 0.;
static double SF_phi0_dn[NG] = { -PI / 3., 0., PI / 3. };
static double SF_phi1_dn[NG] = { 1., -.5, -.5 };
//  static double SF_phi0_up[NG] = {-PI, PI / 3., 2. * PI / 3.};
static double SF_phi0_up[NG] = { -5. * PI / 3., 2. * PI / 3., PI };
static double SF_phi1_up[NG] = { -1., .5, .5 };

#elif NG == 4

static double SF_eta = 0.;
static double SF_phi0_dn[NG] = { -ST * PI / 4., ST *PI / 4. - PI / 2., PI / 2. - ST *PI / 4., ST *PI / 4. };
static double SF_phi1_dn[NG] = { -.5, -.5, .5, .5 };
static double SF_phi0_up[NG] = { -ST * PI / 4. - PI / 2., -PI + ST *PI / 4., PI - ST *PI / 4., PI / 2. + ST *PI / 4. };
static double SF_phi1_up[NG] = { .5, .5, -.5, -.5 };

#endif

void init_gf_SF_BCs(suNg *dn, suNg *up) {
#if (NG > 4)

    error(0, 1, "init_gf_SF_BCs " __FILE__, "SF boundary conditions not defined at this NG");

#else
    int k;

    _suNg_zero(*dn);
    for (k = 0; k < NG; k++) {
        dn->c[(1 + NG) * k] = cos((SF_phi0_dn[k] + SF_phi1_dn[k] * SF_eta) / (GLB_T - 2)) +
                              I * sin((SF_phi0_dn[k] + SF_phi1_dn[k] * SF_eta) / (GLB_T - 2));
    }

    _suNg_zero(*up);
    for (k = 0; k < NG; k++) {
        up->c[(1 + NG) * k] = cos((SF_phi0_up[k] + SF_phi1_up[k] * SF_eta) / (GLB_T - 2)) +
                              I * sin((SF_phi0_up[k] + SF_phi1_up[k] * SF_eta) / (GLB_T - 2));
    }

#if defined(BC_T_SF)
    lprintf("BCS", 0, "SF boundary phases phi0  ( ");
    for (k = 0; k < NG; k++) {
        lprintf("BCS", 0, "%lf ", SF_phi0_dn[k] / 2.);
    }
    lprintf("BCS", 0, ")\n");
    lprintf("BCS", 0, "SF boundary phases phi0'  ( ");
    for (k = 0; k < NG; k++) {
        lprintf("BCS", 0, "%lf ", SF_phi0_up[k] / 2.);
    }
    lprintf("BCS", 0, ")\n");
#else
    lprintf("BCS", 0, "SF boundary phases phi0'  ( ");
    for (k = 0; k < NG; k++) {
        lprintf("BCS", 0, "%lf ", SF_phi0_up[k] / 2.);
    }
    lprintf("BCS", 0, ")\n");
    lprintf("BCS", 0, "SF boundary phases phi1'  ( ");
    for (k = 0; k < NG; k++) {
        lprintf("BCS", 0, "%lf ", SF_phi1_up[k]);
    }
    lprintf("BCS", 0, ")\n");

    lprintf("BCS", 0, "SF boundary phases phi0  ( ");
    for (k = 0; k < NG; k++) {
        lprintf("BCS", 0, "%lf ", SF_phi0_dn[k] / 2.);
    }
    lprintf("BCS", 0, ")\n");
    lprintf("BCS", 0, "SF boundary phases phi1  ( ");
    for (k = 0; k < NG; k++) {
        lprintf("BCS", 0, "%lf ", SF_phi1_dn[k]);
    }
    lprintf("BCS", 0, ")\n");
#endif
#endif
}
#else
void init_gf_SF_BCs(suNg *dn, suNg *up) {
    error(1, 1, "init_gf_SF_BCs", "SF not implemented for real gauge group");
}
#endif

void calc_SF_U(suNg *U, int x0) {
    _suNg_zero(*U);

    for (int k = 0; k < NG; k++) {
        U->c[(1 + NG) * k] = cos(((SF_phi0_dn[k] + SF_phi1_dn[k] * SF_eta) * (GLB_T - 1.0 - x0) +
                                  (SF_phi0_up[k] + SF_phi1_up[k] * SF_eta) * (x0 - 1.0)) /
                                 (GLB_T - 2.0) / (GLB_T - 2.0)) +
                             I * sin(((SF_phi0_dn[k] + SF_phi1_dn[k] * SF_eta) * (GLB_T - 1.0 - x0) +
                                      (SF_phi0_up[k] + SF_phi1_up[k] * SF_eta) * (x0 - 1.0)) /
                                     (GLB_T - 2.0) / (GLB_T - 2));
    }
}
#endif