/// Headerfile for:
/// - suN_exp_group.c
/// - suN_utils.c

#ifndef SUN_MAT_UTILS_H
#define SUN_MAT_UTILS_H

#include "libhr_core.h"

#ifdef __cplusplus
extern "C" {
#endif

// suN_exp_group.c
// SUN exp matrix
visible void suNg_Exp(suNg *u, suNg *Xin); //global function pointer to the correct implementation
visible void ExpX(double dt, suNg_algebra_vector *h, suNg *u);
visible void suNg_Exp_Taylor(suNg *u, suNg *Xin);
visible void normalize_suN(suNg_vector *v);
visible void normalize_suN_flt(suNg_vector_flt *v);

//suN_utils.c
visible void vector_star(suNg_vector *, suNg_vector *);
//visible void project_to_suNg(suNg *u);
visible void project_to_suNg_flt(suNg_flt *u);
#ifndef GAUGE_SON
visible void project_cooling_to_suNg(suNg *g_out, suNg *g_in, int cooling);
#endif
visible void covariant_project_to_suNg(suNg *u);
#ifdef GAUGE_SON
int project_to_suNg_real(suNg *out, suNg *in);
#endif

visible inline __attribute__((flatten)) __attribute__((always_inline)) void project_to_suNg(suNg *u) {
#ifdef GAUGE_SON
    hr_complex norm;
    _suNg_sqnorm(norm, *u);
    if (creal(norm) < 1.e-28) { return; }

    double *v1, *v2;
    int i, j, k;
    double z;
    for (i = 0; i < NG; ++i) {
        v2 = &u->c[i * NG];
        for (j = 0; j < i; ++j) {
            v1 = &u->c[j * NG];
            z = 0;
            for (k = 0; k < NG; ++k) {
                z += v1[k] * v2[k];
            } /*_vector_prod_re_g */
            for (k = 0; k < NG; ++k) {
                v2[k] -= z * v1[k];
            } /*_vector_project_g */
        }
        normalize(v2);
    }
#else
#ifdef WITH_QUATERNIONS
    double norm;
    _suNg_sqnorm(norm, *u);
    if (norm < 1.e-28) { return; }

    _suNg_sqnorm(norm, *u);
    norm = sqrt(0.5 * norm);
    norm = 1. / norm;
    _suNg_mul(*u, norm, *u);

#else
    hr_complex norm;
    _suNg_sqnorm(norm, *u);
    if (creal(norm) < 1.e-28) { return; }

    int i, j;
    suNg_vector *v1, *v2;
    hr_complex z;

    v1 = (suNg_vector *)(u);
    v2 = v1 + 1;
    normalize_suN(v1);
    for (i = 1; i < NG; ++i) {
        for (j = i; j > 0; --j) {
            _vector_prod_g(z, *v1, *v2);
            _vector_project_g(*v2, z, *v1);
            ++v1;
        }
        normalize_suN(v2);
#ifdef WITH_GPU
        // This did not work before without this absolutely cryptic line
        // the compiler now shoes me an error. Might this now
        // work without?
        //memcpy(u->c + NG * i, v2, sizeof(suNg_vector));
#endif
        ++v2;
        v1 = (suNg_vector *)(u);
    }

    det_Cmplx_Ng(&norm, u);
    norm = cpow(norm, -1. / NG);
    _suNg_mul_assign(*u, norm);

#endif
#endif
}

#ifdef __cplusplus
}
#endif

#endif //SUN_MAT_UTILS_H
