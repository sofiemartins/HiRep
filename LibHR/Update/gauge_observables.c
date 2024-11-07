/***************************************************************************\
* Copyright (c) 2008, Claudio Pica                                          *
* All rights reserved.                                                      *
\***************************************************************************/

/*******************************************************************************
*
* File plaquette.c
*
* Routines for the average plaquette
*
*******************************************************************************/

#include "libhr.h"

double plaq(suNg_field *gauge, int ix, int mu, int nu) {
    int iy, iz;
    double p;
    suNg *v1, *v2, *v3, *v4, w1, w2, w3;

    iy = iup(ix, mu);
    iz = iup(ix, nu);

    v1 = _4FIELD_AT(gauge, ix, mu);
    v2 = _4FIELD_AT(gauge, iy, nu);
    v3 = _4FIELD_AT(gauge, iz, mu);
    v4 = _4FIELD_AT(gauge, ix, nu);

    _suNg_times_suNg(w1, (*v1), (*v2));
    _suNg_times_suNg(w2, (*v4), (*v3));
    _suNg_times_suNg_dagger(w3, w1, w2);

    _suNg_trace_re(p, w3);

#ifdef PLAQ_WEIGHTS
    if (plaq_weight == NULL) { return p; }
    return plaq_weight[ix * 16 + mu * 4 + nu] * p;
#else
    return p;
#endif
}

void cplaq(hr_complex *ret, suNg_field *gauge, int ix, int mu, int nu) {
    int iy, iz;
    suNg *v1, *v2, *v3, *v4, w1, w2, w3;
    double tmpre = 0.;
    double tmpim = 0.;

    iy = iup(ix, mu);
    iz = iup(ix, nu);

    v1 = _4FIELD_AT(gauge, ix, mu);
    v2 = _4FIELD_AT(gauge, iy, nu);
    v3 = _4FIELD_AT(gauge, iz, mu);
    v4 = _4FIELD_AT(gauge, ix, nu);

    _suNg_times_suNg(w1, (*v1), (*v2));
    _suNg_times_suNg(w2, (*v4), (*v3));
    _suNg_times_suNg_dagger(w3, w1, w2);

    _suNg_trace_re(tmpre, w3);

#ifndef GAUGE_SON
    _suNg_trace_im(tmpim, w3);
#endif
    *ret = tmpre + I * tmpim;

#ifdef PLAQ_WEIGHTS
    if (plaq_weight != NULL) { *ret *= plaq_weight[ix * 16 + mu * 4 + nu]; }
#endif
}

double local_plaq(suNg_field *gauge, int ix) {
    double pa;
    pa = plaq(gauge, ix, 1, 0);
    pa += plaq(gauge, ix, 2, 0);
    pa += plaq(gauge, ix, 2, 1);
    pa += plaq(gauge, ix, 3, 0);
    pa += plaq(gauge, ix, 3, 1);
    pa += plaq(gauge, ix, 3, 2);
    return pa;
}

double avr_plaquette_cpu() {
    return avr_plaquette_suNg_field(u_gauge);
}

double avr_plaquette_suNg_field_cpu(suNg_field *gauge) {
    static double pa = 0.;

    _OMP_PRAGMA(single) {
        pa = 0.;
    }

#ifdef WITH_NEW_GEOMETRY
    complete_sendrecv_suNg_field(gauge);
#endif

    _PIECE_FOR(&glattice, ixp) {
        if (ixp == glattice.inner_master_pieces) {
            _OMP_PRAGMA(master)
            /* wait for gauge field to be transfered */
            complete_sendrecv_suNg_field(gauge);
            _OMP_PRAGMA(barrier)
        }
        _SITE_FOR_SUM(&glattice, ixp, ix, pa) {
            pa += plaq(gauge, ix, 1, 0);
            pa += plaq(gauge, ix, 2, 0);
            pa += plaq(gauge, ix, 2, 1);
            pa += plaq(gauge, ix, 3, 0);
            pa += plaq(gauge, ix, 3, 1);
            pa += plaq(gauge, ix, 3, 2);
        }
    }

    global_sum(&pa, 1);

#ifdef BC_T_OPEN
    pa /= 6.0 * NG * GLB_VOL3 * (GLB_T - 1);
#elif BC_T_SF
    pa /= 6.0 * NG * GLB_VOL3 * (GLB_T - 2);
#else
    pa /= 6.0 * NG * GLB_VOLUME;
#endif
    return pa;
}

void local_plaquette_cpu(suNg_field *gauge, scalar_field *s) {
#ifdef WITH_NEW_GEOMETRY
    complete_sendrecv_suNg_field(gauge);
#endif

    _PIECE_FOR(&glattice, ixp) {
        if (ixp == glattice.inner_master_pieces) {
            _OMP_PRAGMA(master)
            /* wait for gauge field to be transfered */
            complete_sendrecv_suNg_field(gauge);
            _OMP_PRAGMA(barrier)
        }
        _SITE_FOR(&glattice, ixp, ix) {
            double *pa = _FIELD_AT(s, ix);
            *pa = plaq(gauge, ix, 1, 0);
            *pa += plaq(gauge, ix, 2, 0);
            *pa += plaq(gauge, ix, 2, 1);
            *pa += plaq(gauge, ix, 3, 0);
            *pa += plaq(gauge, ix, 3, 1);
            *pa += plaq(gauge, ix, 3, 2);
        }
    }
}

void avr_plaquette_time_cpu(suNg_field *gauge, double *plaqt, double *plaqs) {
    int ix;
    int tc;
    for (int nt = 0; nt < GLB_T; nt++) {
        plaqt[nt] = plaqs[nt] = 0.0;
    }

    for (int nt = 0; nt < T; nt++) {
        tc = (zerocoord[0] + nt + GLB_T) % GLB_T;
        for (int nx = 0; nx < X; nx++) {
            for (int ny = 0; ny < Y; ny++) {
                for (int nz = 0; nz < Z; nz++) {
                    ix = ipt(nt, nx, ny, nz);
                    plaqt[tc] += plaq(gauge, ix, 1, 0);
                    plaqt[tc] += plaq(gauge, ix, 2, 0);
                    plaqt[tc] += plaq(gauge, ix, 3, 0);
                    plaqs[tc] += plaq(gauge, ix, 2, 1);
                    plaqs[tc] += plaq(gauge, ix, 3, 1);
                    plaqs[tc] += plaq(gauge, ix, 3, 2);
                }
            }
        }
        plaqt[tc] /= 3.0 * NG * GLB_VOL3;
        plaqs[tc] /= 3.0 * NG * GLB_VOL3;
    }

    for (int nt = 0; nt < GLB_T; nt++) {
        global_sum(&plaqt[nt], 1);
        global_sum(&plaqs[nt], 1);
    }
}

void full_plaquette_cpu() {
    return full_plaquette_suNg_field_cpu(u_gauge);
}

void full_plaquette_suNg_field_cpu(suNg_field *gauge) {
    static hr_complex pa[6];
    static hr_complex r0;
    static hr_complex r1;
    static hr_complex r2;
    static hr_complex r3;
    static hr_complex r4;
    static hr_complex r5;

    _OMP_PRAGMA(single) {
        r0 = 0.;
        r1 = 0.;
        r2 = 0.;
        r3 = 0.;
        r4 = 0.;
        r5 = 0.;
    }

#ifdef WITH_NEW_GEOMETRY
    complete_sendrecv_suNg_field(gauge);
#endif
    _PIECE_FOR(&glattice, ixp) {
        if (ixp == glattice.inner_master_pieces) {
            _OMP_PRAGMA(master)
            /* wait for gauge field to be transfered */
            complete_sendrecv_suNg_field(gauge);
            _OMP_PRAGMA(barrier)
        }

        _SITE_FOR_SUM(&glattice, ixp, ix, r0, r1, r2, r3, r4, r5) {
            hr_complex tmp;
            cplaq(&tmp, gauge, ix, 1, 0);
            r0 += tmp;
            cplaq(&tmp, gauge, ix, 2, 0);
            r1 += tmp;
            cplaq(&tmp, gauge, ix, 2, 1);
            r2 += tmp;
            cplaq(&tmp, gauge, ix, 3, 0);
            r3 += tmp;
            cplaq(&tmp, gauge, ix, 3, 1);
            r4 += tmp;
            cplaq(&tmp, gauge, ix, 3, 2);
            r5 += tmp;
        }
    }

    _OMP_PRAGMA(single) {
        pa[0] = r0;
        pa[1] = r1;
        pa[2] = r2;
        pa[3] = r3;
        pa[4] = r4;
        pa[5] = r5;
    }

    global_sum((double *)pa, 12);
    _OMP_PRAGMA(single)
    for (int k = 0; k < 6; k++) {
#ifdef BC_T_OPEN
        pa[k] /= NG * GLB_VOLUME * (GLB_T - 1) / GLB_T;
#else
        pa[k] /= NG * GLB_VOLUME;
#endif
    }

    lprintf("PLAQ", 0, "Plaq(%d,%d) = ( %f , %f )\n", 1, 0, creal(pa[0]), cimag(pa[0]));
    lprintf("PLAQ", 0, "Plaq(%d,%d) = ( %f , %f )\n", 2, 0, creal(pa[1]), cimag(pa[1]));
    lprintf("PLAQ", 0, "Plaq(%d,%d) = ( %f , %f )\n", 2, 1, creal(pa[2]), cimag(pa[2]));
    lprintf("PLAQ", 0, "Plaq(%d,%d) = ( %f , %f )\n", 3, 0, creal(pa[3]), cimag(pa[3]));
    lprintf("PLAQ", 0, "Plaq(%d,%d) = ( %f , %f )\n", 3, 1, creal(pa[4]), cimag(pa[4]));
    lprintf("PLAQ", 0, "Plaq(%d,%d) = ( %f , %f )\n", 3, 2, creal(pa[5]), cimag(pa[5]));
}

void full_momenta(suNg_av_field *momenta) {
    scalar_field *la = alloc_scalar_field(1, &glattice);
#ifdef WITH_FUSE_MASTER_FOR
    _FUSE_MASTER_FOR(&glattice, i) {
        _FUSE_IDX(&glattice, i);
#else
    _MASTER_FOR(&glattice, i) {
#endif
        double a = 0., tmp;
        /* Momenta */
        for (int j = 0; j < 4; ++j) {
            suNg_algebra_vector *cmom = momenta->ptr + coord_to_index(i, j);
            _algebra_vector_sqnorm_g(tmp, *cmom);
            a += tmp; /* this must be positive */
        }
        a *= 0.5 * _FUND_NORM2;
        *_FIELD_AT(la, i) = a;
    }
    static double mom;
    _OMP_PRAGMA(single) {
        mom = 0.;
    }
    _MASTER_FOR_SUM(la->type, i, mom) {
        mom += *_FIELD_AT(la, i);
    }
    lprintf("MOMENTA", 0, "%1.8g\n", mom);
    free_scalar_field(la);
}

void cplaq_wrk(hr_complex *ret, int ix, int mu, int nu) {
    int iy, iz;
    suNg *v1, *v2, *v3, *v4, w1, w2, w3;
#ifdef GAUGE_SON
    double tmpre;
#endif
    iy = iup_wrk(ix, mu);
    iz = iup_wrk(ix, nu);

    v1 = pu_gauge_wrk(ix, mu);
    v2 = pu_gauge_wrk(iy, nu);
    v3 = pu_gauge_wrk(iz, mu);
    v4 = pu_gauge_wrk(ix, nu);

    _suNg_times_suNg(w1, (*v1), (*v2));
    _suNg_times_suNg(w2, (*v4), (*v3));
    _suNg_times_suNg_dagger(w3, w1, w2);

#ifndef GAUGE_SON
    _suNg_trace(*ret, w3);
#else
    _suNg_trace_re(tmpre, w3);
    *ret = tmpre;
#endif

#ifdef PLAQ_WEIGHTS
    if (plaq_weight != NULL) { *ret *= plaq_weight[ix * 16 + mu * 4 + nu]; }
#endif
}

hr_complex avr_plaquette_wrk() {
    static hr_complex pa, tmp;
    suNg_field *_u = u_gauge_wrk();
    start_sendrecv_suNg_field(_u);
#ifdef WITH_NEW_GEOMETRY
    complete_sendrecv_suNg_field(_u);
#endif

    _OMP_PRAGMA(single) {
        pa = tmp = 0.;
    }

    _PIECE_FOR(&glattice, ixp) {
        if (ixp == glattice.inner_master_pieces) {
            _OMP_PRAGMA(master)
            /* wait for gauge field to be transfered */
            complete_sendrecv_suNg_field(_u);
            _OMP_PRAGMA(barrier)
        }
        _SITE_FOR_SUM(&glattice, ixp, ix, pa) {
            cplaq_wrk(&tmp, ix, 1, 0);
            pa += tmp;
            cplaq_wrk(&tmp, ix, 2, 0);
            pa += tmp;
            cplaq_wrk(&tmp, ix, 2, 1);
            pa += tmp;
            cplaq_wrk(&tmp, ix, 3, 0);
            pa += tmp;
            cplaq_wrk(&tmp, ix, 3, 1);
            pa += tmp;
            cplaq_wrk(&tmp, ix, 3, 2);
            pa += tmp;
        }
    }
    global_sum((double *)(&pa), 2);
#ifdef BC_T_OPEN
    pa /= 6.0 * NG * GLB_VOLUME * (GLB_T - 1) / GLB_T;
#else
    pa /= 6.0 * NG * GLB_VOLUME;
#endif
    return pa;
}

double E_cpu(suNg_field *V) {
    double En = 0.;
    _MASTER_FOR_SUM(&glattice, ix, En) {
        double p;
        for (int mu = 0; mu < 4; mu++) {
            for (int nu = mu + 1; nu < 4; nu++) {
                p = plaq(V, ix, mu, nu);
                En += NG - p;
            }
        }
    }
    En *= 2. / ((double)GLB_VOLUME);
    global_sum(&En, 1);
    return En;
}

void E_T_cpu(double *En, suNg_field *V) {
    int gt, t, x, y, z, ix;
    int mu, nu;
    double p;
    for (t = 0; t < 2 * GLB_T; t++) {
        En[t] = 0.;
    }
    for (t = 0; t < T; t++) {
        gt = t + zerocoord[0];
        for (x = 0; x < X; x++) {
            for (y = 0; y < Y; y++) {
                for (z = 0; z < Z; z++) {
                    mu = 0;
                    ix = ipt(t, x, y, z);
                    for (nu = 1; nu < 4; nu++) {
                        p = plaq(V, ix, mu, nu);
                        En[2 * gt] += NG - p;
                    }
                    for (mu = 1; mu < 3; mu++) {
                        for (nu = mu + 1; nu < 4; nu++) {
                            p = plaq(V, ix, mu, nu);
                            En[2 * gt + 1] += NG - p;
                        }
                    }
                }
            }
        }
        En[2 * gt] /= 0.5 * (GLB_VOL3);
        En[2 * gt + 1] /= 0.5 * (GLB_VOL3);
    }
    global_sum(En, 2 * GLB_T);
}

/* This gives F_{\mu\nu}^A */
void clover_F(suNg_algebra_vector *F, suNg_field *V, int ix, int mu, int nu) {
    int iy, iz, iw;
    suNg *v1, *v2, *v3, *v4, w1, w2, w3;
    _suNg_unit(w3);
    _suNg_mul(w3, -4., w3);

    iy = iup(ix, mu);
    iz = iup(ix, nu);

    v1 = _4FIELD_AT(V, ix, mu);
    v2 = _4FIELD_AT(V, iy, nu);
    v3 = _4FIELD_AT(V, iz, mu);
    v4 = _4FIELD_AT(V, ix, nu);

    _suNg_times_suNg(w1, (*v1), (*v2));
    _suNg_times_suNg_dagger(w2, w1, (*v3));
    _suNg_times_suNg_dagger(w1, w2, (*v4));
    _suNg_add_assign(w3, w1);

    iy = idn(ix, mu);
    iz = iup(iy, nu);

    v1 = _4FIELD_AT(V, ix, nu);
    v2 = _4FIELD_AT(V, iz, mu);
    v3 = _4FIELD_AT(V, iy, nu);
    v4 = _4FIELD_AT(V, iy, mu);

    _suNg_times_suNg_dagger(w1, (*v1), (*v2));
    _suNg_times_suNg_dagger(w2, w1, (*v3));
    _suNg_times_suNg(w1, w2, (*v4));
    _suNg_add_assign(w3, w1);

    iy = idn(ix, mu);
    iz = idn(iy, nu);
    iw = idn(ix, nu);

    v1 = _4FIELD_AT(V, iy, mu);
    v2 = _4FIELD_AT(V, iz, nu);
    v3 = _4FIELD_AT(V, iz, mu);
    v4 = _4FIELD_AT(V, iw, nu);

    _suNg_times_suNg(w1, (*v2), (*v1));
    _suNg_dagger_times_suNg(w2, w1, (*v3));
    _suNg_times_suNg(w1, w2, (*v4));
    _suNg_add_assign(w3, w1);

    iy = idn(ix, nu);
    iz = iup(iy, mu);

    v1 = _4FIELD_AT(V, iy, nu);
    v2 = _4FIELD_AT(V, iy, mu);
    v3 = _4FIELD_AT(V, iz, nu);
    v4 = _4FIELD_AT(V, ix, mu);

    _suNg_dagger_times_suNg(w1, (*v1), (*v2));
    _suNg_times_suNg(w2, w1, (*v3));
    _suNg_times_suNg_dagger(w1, w2, (*v4));
    _suNg_add_assign(w3, w1);

    _fund_algebra_project(*F, w3);

    _algebra_vector_mul_g(*F, 1 / 4., *F);
}

double Esym_cpu(suNg_field *V) {
    double En = 0.;
    _MASTER_FOR_SUM(&glattice, ix, En) {
        suNg_algebra_vector clover;
        double p;
        for (int mu = 0; mu < 4; mu++) {
            for (int nu = mu + 1; nu < 4; nu++) {
                clover_F(&clover, V, ix, mu, nu);
                _algebra_vector_sqnorm_g(p, clover);
                En += p;
            }
        }
    }
    En *= _FUND_NORM2 / ((double)GLB_VOLUME);
    global_sum(&En, 1);
    return En;
}

void Esym_T_cpu(double *En, suNg_field *V) {
    int gt, t, x, y, z, ix;
    int mu, nu;
    suNg_algebra_vector clover;
    double p;
    for (t = 0; t < 2 * GLB_T; t++) {
        En[t] = 0.;
    }
    for (t = 0; t < T; t++) {
        gt = t + zerocoord[0];
        for (x = 0; x < X; x++) {
            for (y = 0; y < Y; y++) {
                for (z = 0; z < Z; z++) {
                    mu = 0;
                    ix = ipt(t, x, y, z);
                    for (nu = 1; nu < 4; nu++) {
                        clover_F(&clover, V, ix, mu, nu);
                        _algebra_vector_sqnorm_g(p, clover);
                        En[2 * gt] += p;
                    }
                    for (mu = 1; mu < 4; mu++) {
                        for (nu = mu + 1; nu < 4; nu++) {
                            clover_F(&clover, V, ix, mu, nu);
                            _algebra_vector_sqnorm_g(p, clover);
                            En[2 * gt + 1] += p;
                        }
                    }
                }
            }
        }
        En[2 * gt] *= _FUND_NORM2 / (GLB_VOL3);
        En[2 * gt + 1] *= _FUND_NORM2 / (GLB_VOL3);
    }
    global_sum(En, 2 * GLB_T);
}

/*
  q = 1/(16 \pi^2) \epsilon_{\mu\nu\rho\sigma} \tr F_{\mu\nu} F_{\rho\sigma}
*/

double topo_cpu(suNg_field *V) {
    double TC = 0.;
    int x, y, z, t, ix;
    suNg_algebra_vector F1, F2;
    int t0 = 0, t1 = T;
#if defined(BC_T_OPEN)
    if (COORD[0] == 0) { t0 = 1; }
    if (COORD[0] == NP_T - 1) { t1 = T - 1; }
#elif (defined(BC_T_SF) || defined(BC_T_SF_ROTATED))
    if (COORD[0] == 0) { t0 = 2; }
#endif
    for (t = t0; t < t1; t++) {
        for (x = 0; x < X; x++) {
            for (y = 0; y < Y; y++) {
                for (z = 0; z < Z; z++) {
                    ix = ipt(t, x, y, z);

                    clover_F(&F1, V, ix, 1, 2);
                    clover_F(&F2, V, ix, 0, 3);
                    for (int i = 0; i < NG * NG - 1; i++) {
                        TC += F1.c[i] * F2.c[i];
                    }

                    clover_F(&F1, V, ix, 1, 3);
                    clover_F(&F2, V, ix, 0, 2);
                    for (int i = 0; i < NG * NG - 1; i++) {
                        TC -= F1.c[i] * F2.c[i];
                    }

                    clover_F(&F1, V, ix, 0, 1);
                    clover_F(&F2, V, ix, 2, 3);
                    for (int i = 0; i < NG * NG - 1; i++) {
                        TC += F1.c[i] * F2.c[i];
                    }
                }
            }
        }
    }
    TC *= _FUND_NORM2 / (4. * M_PI * M_PI);
    global_sum(&TC, 1);
    return TC;
}

#ifndef WITH_GPU
double (*E)(suNg_field *V) = E_cpu;
void (*E_T)(double *E, suNg_field *V) = E_T_cpu;
double (*Esym)(suNg_field *V) = Esym_cpu;
void (*Esym_T)(double *E, suNg_field *V) = Esym_T_cpu;
double (*topo)(suNg_field *V) = topo_cpu;
double (*avr_plaquette)() = avr_plaquette_cpu;
double (*avr_plaquette_suNg_field)(suNg_field *gauge) = avr_plaquette_suNg_field_cpu;
void (*full_plaquette)() = full_plaquette_cpu;
void (*full_plaquette_suNg_field)(suNg_field *gauge) = full_plaquette_suNg_field_cpu;
void (*avr_plaquette_time)(suNg_field *gauge, double *plaqt, double *plaqs) = avr_plaquette_time_cpu;
void (*local_plaquette)(suNg_field *gauge, scalar_field *s) = local_plaquette_cpu;
#endif
