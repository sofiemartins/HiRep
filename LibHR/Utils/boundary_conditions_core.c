/***************************************************************************\
* Copyright (c) 2008-2024, Agostino Patella, Claudio Pica, Sofie Martins    *
* All rights reserved.                                                      *
\***************************************************************************/

/***************************************************************************/
/* BOUNDARY CONDITIONS TO BE APPLIED ON THE FUNDAMENTAL GAUGE FIELD        */
/***************************************************************************/

#include "libhr_core.h"
#include "utils.h"
#include "io.h"

#if defined(BC_T_SF) || defined(BC_T_SF_ROTATED)
void gf_SF_BCs_cpu(suNg *dn, suNg *up) {
    int index;
    int ix, iy, iz;
    suNg *u;

    if (COORD[0] == 0) {
        if (T_BORDER > 0) {
            for (ix = 0; ix < X_EXT; ++ix) {
                for (iy = 0; iy < Y_EXT; ++iy) {
                    for (iz = 0; iz < Z_EXT; ++iz) {
                        index = ipt_ext(T_BORDER - 1, ix, iy, iz);
                        if (index != -1) {
                            u = pu_gauge(index, 0);
                            _suNg_unit(*u);
                            u = pu_gauge(index, 1);
                            _suNg_unit(*u);
                            u = pu_gauge(index, 2);
                            _suNg_unit(*u);
                            u = pu_gauge(index, 3);
                            _suNg_unit(*u);
                        }
                    }
                }
            }
        }
        for (ix = 0; ix < X_EXT; ++ix) {
            for (iy = 0; iy < Y_EXT; ++iy) {
                for (iz = 0; iz < Z_EXT; ++iz) {
                    index = ipt_ext(T_BORDER, ix, iy, iz);
                    if (index != -1) {
                        u = pu_gauge(index, 0);
                        _suNg_unit(*u);
                        u = pu_gauge(index, 1);
                        _suNg_unit(*u);
                        u = pu_gauge(index, 2);
                        _suNg_unit(*u);
                        u = pu_gauge(index, 3);
                        _suNg_unit(*u);
                    }
                }
            }
        }
        for (ix = 0; ix < X_EXT; ++ix) {
            for (iy = 0; iy < Y_EXT; ++iy) {
                for (iz = 0; iz < Z_EXT; ++iz) {
                    index = ipt_ext(T_BORDER + 1, ix, iy, iz);
                    if (index != -1) {
                        u = pu_gauge(index, 1);
                        *u = *dn;
                        u = pu_gauge(index, 2);
                        *u = *dn;
                        u = pu_gauge(index, 3);
                        *u = *dn;
                    }
                }
            }
        }
    }
    if (COORD[0] == NP_T - 1) {
        for (ix = 0; ix < X_EXT; ++ix) {
            for (iy = 0; iy < Y_EXT; ++iy) {
                for (iz = 0; iz < Z_EXT; ++iz) {
                    index = ipt_ext(T + T_BORDER - 1, ix, iy, iz);
                    if (index != -1) {
                        u = pu_gauge(index, 0);
                        _suNg_unit(*u);
                        u = pu_gauge(index, 1);
                        *u = *up;
                        u = pu_gauge(index, 2);
                        *u = *up;
                        u = pu_gauge(index, 3);
                        *u = *up;
                    }
                }
            }
        }
        if (T_BORDER > 0) {
            for (ix = 0; ix < X_EXT; ++ix) {
                for (iy = 0; iy < Y_EXT; ++iy) {
                    for (iz = 0; iz < Z_EXT; ++iz) {
                        index = ipt_ext(T + T_BORDER, ix, iy, iz);
                        if (index != -1) {
                            u = pu_gauge(index, 0);
                            _suNg_unit(*u);
                            u = pu_gauge(index, 1);
                            _suNg_unit(*u);
                            u = pu_gauge(index, 2);
                            _suNg_unit(*u);
                            u = pu_gauge(index, 3);
                            _suNg_unit(*u);
                        }
                    }
                }
            }
        }
    }
}

#ifndef WITH_GPU
void (*gf_SF_BCs)(suNg *dn, suNg *up) = gf_SF_BCs_cpu;
#endif
#endif

#if defined(BC_T_SF) || defined(BC_T_SF_ROTATED)
void SF_classical_solution_core_cpu(suNg *U, int it) {
    suNg *u;
    for (int ix = 0; ix < X_EXT; ++ix) {
        for (int iy = 0; iy < Y_EXT; ++iy) {
            for (int iz = 0; iz < Z_EXT; ++iz) {
                int index = ipt_ext(it, ix, iy, iz);
                if (index != -1) {
                    u = pu_gauge(index, 0);
                    _suNg_unit(*u);
                    u = pu_gauge(index, 1);
                    *u = *U;
                    u = pu_gauge(index, 2);
                    *u = *U;
                    u = pu_gauge(index, 3);
                    *u = *U;
                }
            }
        }
    }
}

#ifndef WITH_GPU
void (*SF_classical_solution_core)(suNg *U, int it) = SF_classical_solution_core_cpu;
#endif

void SF_classical_solution(void) {
    double x0;
    suNg U;

    for (int it = 0; it < T_EXT; ++it) {
        x0 = (double)(zerocoord[0] + it - T_BORDER);

        if (x0 >= 1 && x0 < GLB_T - 1) {
            calc_SF_U(&U, x0);
            SF_classical_solution_core(&U, it);
        }
    }

    apply_BCs_on_fundamental_gauge_field();
}

#endif

#ifdef BC_T_OPEN
void gf_open_BCs_cpu() {
    int index;
    int ix, iy, iz;
    suNg *u;

    if (COORD[0] == 0) {
        if (T_BORDER > 0) {
            for (ix = 0; ix < X_EXT; ++ix) {
                for (iy = 0; iy < Y_EXT; ++iy) {
                    for (iz = 0; iz < Z_EXT; ++iz) {
                        index = ipt_ext(T_BORDER - 1, ix, iy, iz);
                        if (index != -1) {
                            u = pu_gauge(index, 0);
                            _suNg_zero(*u);
                            u = pu_gauge(index, 1);
                            _suNg_zero(*u);
                            u = pu_gauge(index, 2);
                            _suNg_zero(*u);
                            u = pu_gauge(index, 3);
                            _suNg_zero(*u);
                        }
                    }
                }
            }
        }
    }
    if (COORD[0] == NP_T - 1) {
        for (ix = 0; ix < X_EXT; ++ix) {
            for (iy = 0; iy < Y_EXT; ++iy) {
                for (iz = 0; iz < Z_EXT; ++iz) {
                    index = ipt_ext(T + T_BORDER - 1, ix, iy, iz);
                    if (index != -1) {
                        u = pu_gauge(index, 0);
                        _suNg_zero(*u);
                    }
                }
            }
        }
        if (T_BORDER > 0) {
            for (ix = 0; ix < X_EXT; ++ix) {
                for (iy = 0; iy < Y_EXT; ++iy) {
                    for (iz = 0; iz < Z_EXT; ++iz) {
                        index = ipt_ext(T + T_BORDER, ix, iy, iz);
                        if (index != -1) {
                            u = pu_gauge(index, 0);
                            _suNg_zero(*u);
                            u = pu_gauge(index, 1);
                            _suNg_zero(*u);
                            u = pu_gauge(index, 2);
                            _suNg_zero(*u);
                            u = pu_gauge(index, 3);
                            _suNg_zero(*u);
                        }
                    }
                }
            }
        }
    }
}

#ifndef WITH_GPU
void (*gf_open_BCs)() = gf_open_BCs_cpu;
#endif
#endif

/***************************************************************************/
/* BOUNDARY CONDITIONS TO BE APPLIED ON THE REPRESENTED GAUGE FIELD        */
/***************************************************************************/

#ifdef BC_T_ANTIPERIODIC
void sp_T_antiperiodic_BCs_cpu() {
    if (COORD[0] == 0) {
        int index;
        int ix, iy, iz;
        suNf *u;
        for (ix = 0; ix < X_EXT; ++ix) {
            for (iy = 0; iy < Y_EXT; ++iy) {
                for (iz = 0; iz < Z_EXT; ++iz) {
                    index = ipt_ext(2 * T_BORDER, ix, iy, iz);
                    if (index != -1) {
                        u = pu_gauge_f(index, 0);
                        _suNf_minus(*u, *u);
                    }
                }
            }
        }
    }
}

#ifndef WITH_GPU
void (*sp_T_antiperiodic_BCs)() = sp_T_antiperiodic_BCs_cpu;
#endif
#endif

#ifdef BC_X_ANTIPERIODIC
void sp_X_antiperiodic_BCs_cpu() {
    if (COORD[1] == 0) {
        int index;
        int it, iy, iz;
        suNf *u;
        for (it = 0; it < T_EXT; ++it) {
            for (iy = 0; iy < Y_EXT; ++iy) {
                for (iz = 0; iz < Z_EXT; ++iz) {
                    index = ipt_ext(it, 2 * X_BORDER, iy, iz);
                    if (index != -1) {
                        u = pu_gauge_f(index, 1);
                        _suNf_minus(*u, *u);
                    }
                }
            }
        }
    }
}

#ifndef WITH_GPU
void (*sp_X_antiperiodic_BCs)() = sp_X_antiperiodic_BCs_cpu;
#endif
#endif

#ifdef BC_Y_ANTIPERIODIC
void sp_Y_antiperiodic_BCs_cpu() {
    if (COORD[2] == 0) {
        int index;
        int ix, it, iz;
        suNf *u;
        for (it = 0; it < T_EXT; ++it) {
            for (ix = 0; ix < X_EXT; ++ix) {
                for (iz = 0; iz < Z_EXT; ++iz) {
                    index = ipt_ext(it, ix, 2 * Y_BORDER, iz);
                    if (index != -1) {
                        u = pu_gauge_f(index, 2);
                        _suNf_minus(*u, *u);
                    }
                }
            }
        }
    }
}

#ifndef WITH_GPU
void (*sp_Y_antiperiodic_BCs)() = sp_Y_antiperiodic_BCs_cpu;
#endif
#endif

#ifdef BC_Z_ANTIPERIODIC
void sp_Z_antiperiodic_BCs_cpu() {
    if (COORD[3] == 0) {
        int index;
        int ix, iy, it;
        suNf *u;
        for (it = 0; it < T_EXT; ++it) {
            for (ix = 0; ix < X_EXT; ++ix) {
                for (iy = 0; iy < Y_EXT; ++iy) {
                    index = ipt_ext(it, ix, iy, 2 * Z_BORDER);
                    if (index != -1) {
                        u = pu_gauge_f(index, 3);
                        _suNf_minus(*u, *u);
                    }
                }
            }
        }
    }
}

#ifndef WITH_GPU
void (*sp_Z_antiperiodic_BCs)() = sp_Z_antiperiodic_BCs_cpu;
#endif
#endif

#ifdef BC_T_SF_ROTATED
void chiSF_ds_BT_cpu(double ds) {
    if (COORD[0] == 0) {
        int index;
        int ix, iy, iz;
        suNf *u;
        for (ix = 0; ix < X_EXT; ++ix) {
            for (iy = 0; iy < Y_EXT; ++iy) {
                for (iz = 0; iz < Z_EXT; ++iz) {
                    index = ipt_ext(T_BORDER + 1, ix, iy, iz);
                    if (index != -1) {
                        u = pu_gauge_f(index, 1);
                        _suNf_mul(*u, ds, *u);
                        u = pu_gauge_f(index, 2);
                        _suNf_mul(*u, ds, *u);
                        u = pu_gauge_f(index, 3);
                        _suNf_mul(*u, ds, *u);
                    }
                }
            }
        }
    }
    if (COORD[0] == NP_T - 1) {
        int index;
        int ix, iy, iz;
        suNf *u;
        for (ix = 0; ix < X_EXT; ++ix) {
            for (iy = 0; iy < Y_EXT; ++iy) {
                for (iz = 0; iz < Z_EXT; ++iz) {
                    index = ipt_ext(T + T_BORDER - 1, ix, iy, iz);
                    if (index != -1) {
                        u = pu_gauge_f(index, 1);
                        _suNf_mul(*u, ds, *u);
                        u = pu_gauge_f(index, 2);
                        _suNf_mul(*u, ds, *u);
                        u = pu_gauge_f(index, 3);
                        _suNf_mul(*u, ds, *u);
                    }
                }
            }
        }
    }
}

#ifndef WITH_GPU
void (*chiSF_ds_BT)(double ds) = chiSF_ds_BT_cpu;
#endif
#endif

/***************************************************************************/
/* BOUNDARY CONDITIONS TO BE APPLIED ON THE CLOVER TERM                    */
/***************************************************************************/

#if (defined(WITH_EXPCLOVER) || defined(WITH_CLOVER)) && (defined(BC_T_SF) || defined(BC_T_SF_ROTATED))
void cl_SF_BCs_cpu(clover_term *cl) {
    int index;
    suNfc u;
    _suNfc_zero(u);

    // These should reflect the boundary conditions imposed on the spinor fields
    if (COORD[0] == 0) {
        for (int ix = 0; ix < X_EXT; ix++) {
            for (int iy = 0; iy < Y_EXT; iy++) {
                for (int iz = 0; iz < Z_EXT; iz++) {
                    if (T_BORDER > 0) {
                        index = ipt_ext(T_BORDER - 1, ix, iy, iz);
                        if (index != -1) {
                            for (int mu = 0; mu < 4; mu++) {
                                *_4FIELD_AT(cl, index, mu) = u;
                            }
                        }
                    }
                    index = ipt_ext(T_BORDER, ix, iy, iz);
                    if (index != -1) {
                        for (int mu = 0; mu < 4; mu++) {
                            *_4FIELD_AT(cl, index, mu) = u;
                        }
                    }
                    index = ipt_ext(T_BORDER + 1, ix, iy, iz);
                    if (index != -1) {
                        for (int mu = 0; mu < 4; mu++) {
                            *_4FIELD_AT(cl, index, mu) = u;
                        }
                    }
                }
            }
        }
    }
}

#ifndef WITH_GPU
void (*cl_SF_BCs)(clover_term *cl) = cl_SF_BCs_cpu;
#endif
#endif

#if (defined(WITH_EXPCLOVER) || defined(WITH_CLOVER)) && defined(BC_T_OPEN)
void cl_open_BCs_cpu(clover_term *cl) {
    int index;
    suNfc u;
    _suNfc_zero(u);

    // These should reflect the boundary conditions imposed on the spinor fields
    if (COORD[0] == 0) {
        for (int ix = 0; ix < X_EXT; ix++) {
            for (int iy = 0; iy < Y_EXT; iy++) {
                for (int iz = 0; iz < Z_EXT; iz++) {
                    if (T_BORDER > 0) {
                        index = ipt_ext(T_BORDER - 1, ix, iy, iz);
                        if (index != -1) {
                            for (int mu = 0; mu < 4; mu++) {
                                *_4FIELD_AT(cl, index, mu) = u;
                            }
                        }
                    }
                    index = ipt_ext(T_BORDER, ix, iy, iz);
                    if (index != -1) {
                        for (int mu = 0; mu < 4; mu++) {
                            *_4FIELD_AT(cl, index, mu) = u;
                        }
                    }
                }
            }
        }
    }
    if (COORD[0] == NP_T - 1) {
        for (int ix = 0; ix < X_EXT; ix++) {
            for (int iy = 0; iy < Y_EXT; iy++) {
                for (int iz = 0; iz < Z_EXT; iz++) {
                    index = ipt_ext(T + T_BORDER - 1, ix, iy, iz);
                    if (index != -1) {
                        for (int mu = 0; mu < 4; mu++) {
                            *_4FIELD_AT(cl, index, mu) = u;
                        }
                    }
                    if (T_BORDER > 0) {
                        index = ipt_ext(T + T_BORDER, ix, iy, iz);
                        if (index != -1) {
                            for (int mu = 0; mu < 4; mu++) {
                                *_4FIELD_AT(cl, index, mu) = u;
                            }
                        }
                    }
                }
            }
        }
    }
}

#ifndef WITH_GPU
void (*cl_open_BCs)(clover_term *cl) = cl_open_BCs_cpu;
#endif
#endif

/***************************************************************************/
/* BOUNDARY CONDITIONS TO BE APPLIED ON THE MOMENTUM FIELDS                */
/***************************************************************************/

#if defined(BC_T_SF) || defined(BC_T_SF_ROTATED)
void mf_Dirichlet_BCs_cpu(suNg_av_field *force) {
    int ix, iy, iz, index;

    if (COORD[0] == 0) {
        if (T_BORDER > 0) {
            for (ix = 0; ix < X_EXT; ++ix) {
                for (iy = 0; iy < Y_EXT; ++iy) {
                    for (iz = 0; iz < Z_EXT; ++iz) {
                        index = ipt_ext(T_BORDER - 1, ix, iy, iz);
                        if (index != -1) {
                            _algebra_vector_zero_g(*_4FIELD_AT(force, index, 0));
                            _algebra_vector_zero_g(*_4FIELD_AT(force, index, 1));
                            _algebra_vector_zero_g(*_4FIELD_AT(force, index, 2));
                            _algebra_vector_zero_g(*_4FIELD_AT(force, index, 3));
                        }
                    }
                }
            }
        }
        for (ix = 0; ix < X_EXT; ++ix) {
            for (iy = 0; iy < Y_EXT; ++iy) {
                for (iz = 0; iz < Z_EXT; ++iz) {
                    index = ipt_ext(T_BORDER, ix, iy, iz);
                    if (index != -1) {
                        _algebra_vector_zero_g(*_4FIELD_AT(force, index, 0));
                        _algebra_vector_zero_g(*_4FIELD_AT(force, index, 1));
                        _algebra_vector_zero_g(*_4FIELD_AT(force, index, 2));
                        _algebra_vector_zero_g(*_4FIELD_AT(force, index, 3));
                    }

                    index = ipt_ext(T_BORDER + 1, ix, iy, iz);
                    if (index != -1) {
                        _algebra_vector_zero_g(*_4FIELD_AT(force, index, 1));
                        _algebra_vector_zero_g(*_4FIELD_AT(force, index, 2));
                        _algebra_vector_zero_g(*_4FIELD_AT(force, index, 3));
                    }
                }
            }
        }
    }

    if (COORD[0] == NP_T - 1) {
        for (ix = 0; ix < X_EXT; ++ix) {
            for (iy = 0; iy < Y_EXT; ++iy) {
                for (iz = 0; iz < Z_EXT; ++iz) {
                    index = ipt_ext(T + T_BORDER - 1, ix, iy, iz);
                    if (index != -1) {
                        _algebra_vector_zero_g(*_4FIELD_AT(force, index, 0));
                        _algebra_vector_zero_g(*_4FIELD_AT(force, index, 1));
                        _algebra_vector_zero_g(*_4FIELD_AT(force, index, 2));
                        _algebra_vector_zero_g(*_4FIELD_AT(force, index, 3));
                    }
                }
            }
        }
        if (T_BORDER > 0) {
            for (ix = 0; ix < X_EXT; ++ix) {
                for (iy = 0; iy < Y_EXT; ++iy) {
                    for (iz = 0; iz < Z_EXT; ++iz) {
                        index = ipt_ext(T + T_BORDER, ix, iy, iz);
                        if (index != -1) {
                            _algebra_vector_zero_g(*_4FIELD_AT(force, index, 0));
                            _algebra_vector_zero_g(*_4FIELD_AT(force, index, 1));
                            _algebra_vector_zero_g(*_4FIELD_AT(force, index, 2));
                            _algebra_vector_zero_g(*_4FIELD_AT(force, index, 3));
                        }
                    }
                }
            }
        }
    }
}

#ifndef WITH_GPU
void (*mf_Dirichlet_BCs)(suNg_av_field *force) = mf_Dirichlet_BCs_cpu;
#endif
#endif

#ifdef BC_T_OPEN
void mf_open_BCs_cpu(suNg_av_field *force) {
    int ix, iy, iz, index;

    if (COORD[0] == 0) {
        if (T_BORDER > 0) {
            for (ix = 0; ix < X_EXT; ++ix) {
                for (iy = 0; iy < Y_EXT; ++iy) {
                    for (iz = 0; iz < Z_EXT; ++iz) {
                        index = ipt_ext(T_BORDER - 1, ix, iy, iz);
                        if (index != -1) {
                            _algebra_vector_zero_g(*_4FIELD_AT(force, index, 0));
                            _algebra_vector_zero_g(*_4FIELD_AT(force, index, 1));
                            _algebra_vector_zero_g(*_4FIELD_AT(force, index, 2));
                            _algebra_vector_zero_g(*_4FIELD_AT(force, index, 3));
                        }
                    }
                }
            }
        }
    }

    if (COORD[0] == NP_T - 1) {
        for (ix = 0; ix < X_EXT; ++ix) {
            for (iy = 0; iy < Y_EXT; ++iy) {
                for (iz = 0; iz < Z_EXT; ++iz) {
                    index = ipt_ext(T + T_BORDER - 1, ix, iy, iz);
                    if (index != -1) { _algebra_vector_zero_g(*_4FIELD_AT(force, index, 0)); }
                }
            }
        }
        if (T_BORDER > 0) {
            for (ix = 0; ix < X_EXT; ++ix) {
                for (iy = 0; iy < Y_EXT; ++iy) {
                    for (iz = 0; iz < Z_EXT; ++iz) {
                        index = ipt_ext(T + T_BORDER, ix, iy, iz);
                        if (index != -1) {
                            _algebra_vector_zero_g(*_4FIELD_AT(force, index, 0));
                            _algebra_vector_zero_g(*_4FIELD_AT(force, index, 1));
                            _algebra_vector_zero_g(*_4FIELD_AT(force, index, 2));
                            _algebra_vector_zero_g(*_4FIELD_AT(force, index, 3));
                        }
                    }
                }
            }
        }
    }
}

#ifndef WITH_GPU
void (*mf_open_BCs)(suNg_av_field *force) = mf_open_BCs_cpu;
#endif
#endif

/***************************************************************************/
/* BOUNDARY CONDITIONS TO BE APPLIED ON THE SPINOR FIELDS                  */
/***************************************************************************/

#if defined(BC_T_SF)
void sf_Dirichlet_BCs_cpu(spinor_field *sp) {
    int ix, iy, iz, index;
    if (COORD[0] == 0) {
        for (ix = 0; ix < X_EXT; ++ix) {
            for (iy = 0; iy < Y_EXT; ++iy) {
                for (iz = 0; iz < Z_EXT; ++iz) {
                    index = ipt_ext(T_BORDER, ix, iy, iz);
                    if (index != -1 && sp->type->master_shift <= index &&
                        sp->type->master_shift + sp->type->gsize_spinor > index) {
                        _spinor_zero_f(*_FIELD_AT(sp, index));
                    }
                    index = ipt_ext(T_BORDER + 1, ix, iy, iz);
                    if (index != -1 && sp->type->master_shift <= index &&
                        sp->type->master_shift + sp->type->gsize_spinor > index) {
                        _spinor_zero_f(*_FIELD_AT(sp, index));
                    }
                }
            }
        }
    }
    if (COORD[0] == NP_T - 1) {
        for (ix = 0; ix < X_EXT; ++ix) {
            for (iy = 0; iy < Y_EXT; ++iy) {
                for (iz = 0; iz < Z_EXT; ++iz) {
                    index = ipt_ext(T + T_BORDER - 1, ix, iy, iz);
                    if (index != -1 && sp->type->master_shift <= index &&
                        sp->type->master_shift + sp->type->gsize_spinor > index) {
                        _spinor_zero_f(*_FIELD_AT(sp, index));
                    }
                }
            }
        }
    }
}

#ifndef WITH_GPU
void (*sf_Dirichlet_BCs)(spinor_field *sp) = sf_Dirichlet_BCs_cpu;
#endif
#endif

#if defined(BC_T_SF)
void sf_Dirichlet_BCs_flt_cpu(spinor_field_flt *sp) {
    int ix, iy, iz, index;
    if (COORD[0] == 0) {
        for (ix = 0; ix < X_EXT; ++ix) {
            for (iy = 0; iy < Y_EXT; ++iy) {
                for (iz = 0; iz < Z_EXT; ++iz) {
                    index = ipt_ext(T_BORDER, ix, iy, iz);
                    if (index != -1 && sp->type->master_shift <= index &&
                        sp->type->master_shift + sp->type->gsize_spinor > index) {
                        _spinor_zero_f(*_FIELD_AT(sp, index));
                    }
                    index = ipt_ext(T_BORDER + 1, ix, iy, iz);
                    if (index != -1 && sp->type->master_shift <= index &&
                        sp->type->master_shift + sp->type->gsize_spinor > index) {
                        _spinor_zero_f(*_FIELD_AT(sp, index));
                    }
                }
            }
        }
    }
    if (COORD[0] == NP_T - 1) {
        for (ix = 0; ix < X_EXT; ++ix) {
            for (iy = 0; iy < Y_EXT; ++iy) {
                for (iz = 0; iz < Z_EXT; ++iz) {
                    index = ipt_ext(T + T_BORDER - 1, ix, iy, iz);
                    if (index != -1 && sp->type->master_shift <= index &&
                        sp->type->master_shift + sp->type->gsize_spinor > index) {
                        _spinor_zero_f(*_FIELD_AT(sp, index));
                    }
                }
            }
        }
    }
}

#ifndef WITH_GPU
void (*sf_Dirichlet_BCs_flt)(spinor_field_flt *sp) = sf_Dirichlet_BCs_flt_cpu;
#endif
#endif

#if defined(BC_T_SF_ROTATED)
void sf_open_BCs_cpu(spinor_field *sp) {
    int ix, iy, iz, index;
    if (COORD[0] == 0) {
        for (ix = 0; ix < X_EXT; ++ix) {
            for (iy = 0; iy < Y_EXT; ++iy) {
                for (iz = 0; iz < Z_EXT; ++iz) {
                    index = ipt_ext(T_BORDER, ix, iy, iz);
                    if (index != -1 && sp->type->master_shift <= index &&
                        sp->type->master_shift + sp->type->gsize_spinor > index) {
                        _spinor_zero_f(*_FIELD_AT(sp, index));
                    }
                }
            }
        }
    }
}

#ifndef WITH_GPU
void (*sf_open_BCs)(spinor_field *sp) = sf_open_BCs_cpu;
#endif
#endif

#if defined(BC_T_SF_ROTATED)
void sf_open_BCs_flt_cpu(spinor_field_flt *sp) {
    int ix, iy, iz, index;
    if (COORD[0] == 0) {
        for (ix = 0; ix < X_EXT; ++ix) {
            for (iy = 0; iy < Y_EXT; ++iy) {
                for (iz = 0; iz < Z_EXT; ++iz) {
                    index = ipt_ext(T_BORDER, ix, iy, iz);
                    if (index != -1 && sp->type->master_shift <= index &&
                        sp->type->master_shift + sp->type->gsize_spinor > index) {
                        _spinor_zero_f(*_FIELD_AT(sp, index));
                    }
                }
            }
        }
    }
}

#ifndef WITH_GPU
void (*sf_open_BCs_flt)(spinor_field_flt *sp) = sf_open_BCs_flt_cpu;
#endif
#endif

#ifdef BC_T_OPEN
void sf_open_v2_BCs_cpu(spinor_field *sf) {
    int index;
    suNf_spinor u;
    _spinor_zero_f(u);

    // These should reflect the boundary conditions imposed on the clover field
    if (COORD[0] == 0) {
        for (int ix = 0; ix < X_EXT; ix++) {
            for (int iy = 0; iy < Y_EXT; iy++) {
                for (int iz = 0; iz < Z_EXT; iz++) {
                    if (T_BORDER > 0) {
                        index = ipt_ext(T_BORDER - 1, ix, iy, iz);
                        if (index != -1 && sf->type->master_shift <= index &&
                            sf->type->master_shift + sf->type->gsize_spinor > index) {
                            *_FIELD_AT(sf, index) = u;
                        }
                    }
                    index = ipt_ext(T_BORDER, ix, iy, iz);
                    if (index != -1 && sf->type->master_shift <= index &&
                        sf->type->master_shift + sf->type->gsize_spinor > index) {
                        *_FIELD_AT(sf, index) = u;
                    }
                }
            }
        }
    }
    if (COORD[0] == NP_T - 1) {
        for (int ix = 0; ix < X_EXT; ix++) {
            for (int iy = 0; iy < Y_EXT; iy++) {
                for (int iz = 0; iz < Z_EXT; iz++) {
                    index = ipt_ext(T + T_BORDER - 1, ix, iy, iz);
                    if (index != -1 && sf->type->master_shift <= index &&
                        sf->type->master_shift + sf->type->gsize_spinor > index) {
                        *_FIELD_AT(sf, index) = u;
                    }
                    if (T_BORDER > 0) {
                        index = ipt_ext(T + T_BORDER, ix, iy, iz);
                        if (index != -1 && sf->type->master_shift <= index &&
                            sf->type->master_shift + sf->type->gsize_spinor > index) {
                            *_FIELD_AT(sf, index) = u;
                        }
                    }
                }
            }
        }
    }
}

#ifndef WITH_GPU
void (*sf_open_v2_BCs)(spinor_field *sf) = sf_open_v2_BCs_cpu;
#endif
#endif

#ifdef BC_T_OPEN
void sf_open_v2_BCs_flt_cpu(spinor_field_flt *sf) {
    int index;
    suNf_spinor_flt u;
    _spinor_zero_f(u);

    // These should reflect the boundary conditions imposed on the clover field
    if (COORD[0] == 0) {
        for (int ix = 0; ix < X_EXT; ix++) {
            for (int iy = 0; iy < Y_EXT; iy++) {
                for (int iz = 0; iz < Z_EXT; iz++) {
                    if (T_BORDER > 0) {
                        index = ipt_ext(T_BORDER - 1, ix, iy, iz);
                        if (index != -1 && sf->type->master_shift <= index &&
                            sf->type->master_shift + sf->type->gsize_spinor > index) {
                            *_FIELD_AT(sf, index) = u;
                        }
                    }
                    index = ipt_ext(T_BORDER, ix, iy, iz);
                    if (index != -1 && sf->type->master_shift <= index &&
                        sf->type->master_shift + sf->type->gsize_spinor > index) {
                        *_FIELD_AT(sf, index) = u;
                    }
                }
            }
        }
    }
    if (COORD[0] == NP_T - 1) {
        for (int ix = 0; ix < X_EXT; ix++) {
            for (int iy = 0; iy < Y_EXT; iy++) {
                for (int iz = 0; iz < Z_EXT; iz++) {
                    index = ipt_ext(T + T_BORDER - 1, ix, iy, iz);
                    if (index != -1 && sf->type->master_shift <= index &&
                        sf->type->master_shift + sf->type->gsize_spinor > index) {
                        *_FIELD_AT(sf, index) = u;
                    }
                    if (T_BORDER > 0) {
                        index = ipt_ext(T + T_BORDER, ix, iy, iz);
                        if (index != -1 && sf->type->master_shift <= index &&
                            sf->type->master_shift + sf->type->gsize_spinor > index) {
                            *_FIELD_AT(sf, index) = u;
                        }
                    }
                }
            }
        }
    }
}

#ifndef WITH_GPU
void (*sf_open_v2_BCs_flt)(spinor_field_flt *sp) = sf_open_v2_BCs_flt_cpu;
#endif
#endif

/***************************************************************************/
/* BOUNDARY CONDITIONS TO BE APPLIED IN THE WILSON ACTION                  */
/***************************************************************************/

#ifdef PLAQ_WEIGHTS

#ifdef GAUGE_SPATIAL_TWIST
void init_plaq_twisted_BCs() {
    error(plaq_weight == NULL, 1, __func__, "Structure plaq_weight not initialized yet");

    int loc[4] = { T, X, Y, Z };
    int mu, nu, rho, x[4], index;

    for (mu = 1; mu < 3; mu++) {
        for (nu = mu + 1; nu < 4; nu++) {
            rho = 6 - mu - nu;
            x[mu] = 1;
            x[nu] = 1;
            if (COORD[mu] == 0 && COORD[nu] == 0) {
                for (x[0] = 0; x[0] < T; x[0]++) {
                    for (x[rho] = 0; x[rho] < loc[rho]; x[rho]++) {
                        index = ipt(x[0], x[1], x[2], x[3]);
                        plaq_weight[index * 16 + mu * 4 + nu] *= -1.;
                        plaq_weight[index * 16 + nu * 4 + mu] *= -1.; /*IF COMPLEX, THE WEIGHT SHOULD BE C.C.*/
                    }
                }
            }
        }
    }
#ifdef WITH_GPU
    CHECK_CUDA(cudaMemcpy(plaq_weight_gpu, plaq_weight, 16 * glattice.gsize_gauge * sizeof(double), cudaMemcpyHostToDevice));
#endif

    lprintf("BCS", 0, "Twisted BCs. Dirac strings intersecting at ( X , Y , Z ) = ( 1 , 1 , 1 )\n");
}
#endif //GAUGE_SPATIAL_TWIST

void init_plaq_open_BCs(double *lplaq_weight, double *lrect_weight, double ct, double cs) {
    error(lplaq_weight == NULL, 1, __func__, "Structure plaq_weight not initialized yet");

    int mu, nu, ix, iy, iz, index;

    if (COORD[0] == 0) {
        if (T_BORDER > 0) {
            for (ix = 0; ix < X_EXT; ++ix) {
                for (iy = 0; iy < Y_EXT; ++iy) {
                    for (iz = 0; iz < Z_EXT; ++iz) {
                        index = ipt_ext(T_BORDER - 1, ix, iy, iz);
                        if (index != -1) {
                            mu = 0;
                            for (nu = mu + 1; nu < 4; nu++) {
                                lplaq_weight[index * 16 + mu * 4 + nu] = 0;
                                lplaq_weight[index * 16 + nu * 4 + mu] = 0;
                                if (lrect_weight != NULL) { lrect_weight[index * 16 + mu * 4 + nu] = 0; }
                                if (lrect_weight != NULL) { lrect_weight[index * 16 + nu * 4 + mu] = 0; }
                            }
                            for (mu = 1; mu < 3; mu++) {
                                for (nu = mu + 1; nu < 4; nu++) {
                                    lplaq_weight[index * 16 + mu * 4 + nu] = 0.5 * cs;
                                    lplaq_weight[index * 16 + nu * 4 + mu] = 0.5 * cs;
                                    if (lrect_weight != NULL) { lrect_weight[index * 16 + mu * 4 + nu] = 0.5 * cs; }
                                    if (lrect_weight != NULL) { lrect_weight[index * 16 + nu * 4 + mu] = 0.5 * cs; }
                                }
                            }
                        }
                    }
                }
            }
        }

        for (ix = 0; ix < X_EXT; ++ix) {
            for (iy = 0; iy < Y_EXT; ++iy) {
                for (iz = 0; iz < Z_EXT; ++iz) {
                    index = ipt_ext(T_BORDER, ix, iy, iz);
                    if (index != -1) {
                        mu = 0;
                        for (nu = mu + 1; nu < 4; nu++) {
                            lplaq_weight[index * 16 + mu * 4 + nu] = ct;
                            lplaq_weight[index * 16 + nu * 4 + mu] = ct;
                        }
                        for (mu = 1; mu < 3; mu++) {
                            for (nu = mu + 1; nu < 4; nu++) {
                                lplaq_weight[index * 16 + mu * 4 + nu] = 0.5 * cs;
                                lplaq_weight[index * 16 + nu * 4 + mu] = 0.5 * cs;
                                if (lrect_weight != NULL) { lrect_weight[index * 16 + mu * 4 + nu] = 0.5 * cs; }
                                if (lrect_weight != NULL) { lrect_weight[index * 16 + nu * 4 + mu] = 0.5 * cs; }
                            }
                        }
                    }
                }
            }
        }
    }

    if (COORD[0] == NP_T - 1) {
        if (T_BORDER > 0) {
            for (ix = 0; ix < X_EXT; ++ix) {
                for (iy = 0; iy < Y_EXT; ++iy) {
                    for (iz = 0; iz < Z_EXT; ++iz) {
                        index = ipt_ext(T + T_BORDER, ix, iy, iz);
                        if (index != -1) {
                            mu = 0;
                            for (nu = mu + 1; nu < 4; nu++) {
                                lplaq_weight[index * 16 + mu * 4 + nu] = ct;
                                lplaq_weight[index * 16 + nu * 4 + mu] = ct;
                            }
                            for (mu = 1; mu < 3; mu++) {
                                for (nu = mu + 1; nu < 4; nu++) {
                                    lplaq_weight[index * 16 + mu * 4 + nu] = 0.5 * cs;
                                    lplaq_weight[index * 16 + nu * 4 + mu] = 0.5 * cs;
                                    if (lrect_weight != NULL) { lrect_weight[index * 16 + mu * 4 + nu] = 0.5 * cs; }
                                    if (lrect_weight != NULL) { lrect_weight[index * 16 + nu * 4 + mu] = 0.5 * cs; }
                                }
                            }
                        }
                    }
                }
            }
        }

        for (ix = 0; ix < X_EXT; ++ix) {
            for (iy = 0; iy < Y_EXT; ++iy) {
                for (iz = 0; iz < Z_EXT; ++iz) {
                    index = ipt_ext(T + T_BORDER - 2, ix, iy, iz);
                    if (index != -1) {
                        mu = 0;
                        for (nu = mu + 1; nu < 4; nu++) {
                            lplaq_weight[index * 16 + mu * 4 + nu] = ct;
                            lplaq_weight[index * 16 + nu * 4 + mu] = ct;
                            if (lrect_weight != NULL) { lrect_weight[index * 16 + nu * 4 + mu] = 0; }
                        }
                    }

                    index = ipt_ext(T + T_BORDER - 1, ix, iy, iz);
                    if (index != -1) {
                        mu = 0;
                        for (nu = mu + 1; nu < 4; nu++) {
                            lplaq_weight[index * 16 + mu * 4 + nu] = 0;
                            lplaq_weight[index * 16 + nu * 4 + mu] = 0;
                            if (lrect_weight != NULL) { lrect_weight[index * 16 + mu * 4 + nu] = 0; }
                            if (lrect_weight != NULL) { lrect_weight[index * 16 + nu * 4 + mu] = 0; }
                        }
                        for (mu = 1; mu < 3; mu++) {
                            for (nu = mu + 1; nu < 4; nu++) {
                                lplaq_weight[index * 16 + mu * 4 + nu] = 0.5 * cs;
                                lplaq_weight[index * 16 + nu * 4 + mu] = 0.5 * cs;
                                if (lrect_weight != NULL) { lrect_weight[index * 16 + mu * 4 + nu] = 0.5 * cs; }
                                if (lrect_weight != NULL) { lrect_weight[index * 16 + nu * 4 + mu] = 0.5 * cs; }
                            }
                        }
                    }
                }
            }
        }
    }
}

#if defined(BC_T_SF) || defined(BC_T_SF_ROTATED)

void init_plaq_SF_BCs(double ct) {
    error(plaq_weight == NULL, 1, __func__, "Structure plaq_weight not initialized yet");
    int mu, nu, ix, iy, iz, index;

    if (COORD[0] == 0) {
        if (T_BORDER > 0) {
            for (ix = 0; ix < X_EXT; ++ix) {
                for (iy = 0; iy < Y_EXT; ++iy) {
                    for (iz = 0; iz < Z_EXT; ++iz) {
                        index = ipt_ext(T_BORDER - 1, ix, iy, iz);
                        if (index != -1) {
                            for (mu = 0; mu < 3; mu++) {
                                for (nu = mu + 1; nu < 4; nu++) {
                                    plaq_weight[index * 16 + mu * 4 + nu] = 0.0; //0.5 * cs;
                                    plaq_weight[index * 16 + nu * 4 + mu] = 0.0;
                                }
                            }
                        }
                    }
                }
            }
        }

        for (ix = 0; ix < X_EXT; ++ix) {
            for (iy = 0; iy < Y_EXT; ++iy) {
                for (iz = 0; iz < Z_EXT; ++iz) {
                    index = ipt_ext(T_BORDER, ix, iy, iz);
                    if (index != -1) {
                        for (mu = 0; mu < 3; mu++) {
                            for (nu = mu + 1; nu < 4; nu++) {
                                plaq_weight[index * 16 + mu * 4 + nu] = 0.0;
                                plaq_weight[index * 16 + nu * 4 + mu] = 0.0;
                            }
                        }
                    }
                    index = ipt_ext(T_BORDER + 1, ix, iy, iz);
                    if (index != -1) {
                        mu = 0;
                        for (nu = mu + 1; nu < 4; nu++) {
                            plaq_weight[index * 16 + mu * 4 + nu] = ct;
                            plaq_weight[index * 16 + nu * 4 + mu] = ct;
                        }

                        for (mu = 1; mu < 3; mu++) {
                            for (nu = mu + 1; nu < 4; nu++) {
                                plaq_weight[index * 16 + mu * 4 + nu] = 0.0;
                                plaq_weight[index * 16 + nu * 4 + mu] = 0.0;
                            }
                        }
                    }
                }
            }
        }
    }

    if (COORD[0] == NP_T - 1) {
        if (T_BORDER > 0) {
            for (ix = 0; ix < X_EXT; ++ix) {
                for (iy = 0; iy < Y_EXT; ++iy) {
                    for (iz = 0; iz < Z_EXT; ++iz) {
                        index = ipt_ext(T + T_BORDER, ix, iy, iz);
                        if (index != -1) {
                            for (mu = 0; mu < 3; mu++) {
                                for (nu = mu + 1; nu < 4; nu++) {
                                    plaq_weight[index * 16 + mu * 4 + nu] = 0.0;
                                    plaq_weight[index * 16 + nu * 4 + mu] = 0.0;
                                }
                            }
                        }
                    }
                }
            }
        }

        for (ix = 0; ix < X_EXT; ++ix) {
            for (iy = 0; iy < Y_EXT; ++iy) {
                for (iz = 0; iz < Z_EXT; ++iz) {
                    index = ipt_ext(T + T_BORDER - 2, ix, iy, iz);
                    if (index != -1) {
                        mu = 0;
                        for (nu = mu + 1; nu < 4; nu++) {
                            plaq_weight[index * 16 + mu * 4 + nu] = ct;
                            plaq_weight[index * 16 + nu * 4 + mu] = ct;
                        }
                    }

                    index = ipt_ext(T + T_BORDER - 1, ix, iy, iz);
                    if (index != -1) {
                        for (mu = 0; mu < 3; mu++) {
                            for (nu = mu + 1; nu < 4; nu++) {
                                plaq_weight[index * 16 + mu * 4 + nu] = 0.0;
                                plaq_weight[index * 16 + nu * 4 + mu] = 0.0;
                            }
                        }
                    }
                }
            }
        }
    }
#ifdef WITH_GPU
    CHECK_CUDA(cudaMemcpy(plaq_weight_gpu, plaq_weight, 16 * glattice.gsize_gauge * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(rect_weight_gpu, rect_weight, 16 * glattice.gsize_gauge * sizeof(double), cudaMemcpyHostToDevice));
#endif
}
#endif
#endif
