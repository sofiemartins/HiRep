/***************************************************************************\
* Copyright (c) 2008, Claudio Pica                                          *
* All rights reserved.                                                      *
\***************************************************************************/

/*******************************************************************************
 *
 * File su3_utils.c
 *
 * Functions to project to SU(3)
 *
 *******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "utils.h"
#include "suN.h"
#include "representation.h"
#include "logger.h"

void vector_star(suNg_vector *v1, suNg_vector *v2)
{
  for (int i = 0; i < NG; i++)
  {
    _complex_star(v1->c[i], v2->c[i]);
  }
}

#ifdef GAUGE_SON
static void normalize(double *v)
{
  double fact = 0;
  int i;
  for (i = 0; i < NG; ++i)
  {
    fact += v[i] * v[i];
  }
  fact = 1.0 / sqrt(fact);
  for (i = 0; i < NG; ++i)
  {
    v[i] *= fact;
  }
}
static void normalize_flt(float *v)
{
  float fact = 0;
  int i;
  for (i = 0; i < NG; ++i)
  {
    fact += v[i] * v[i];
  }
  fact = 1.0 / sqrt(fact);
  for (i = 0; i < NG; ++i)
  {
    v[i] *= fact;
  }
}
#elif !defined(WITH_QUATERNIONS)
static void normalize(suNg_vector *v)
{
  double fact;
  _vector_prod_re_g(fact, *v, *v);
  fact = 1.0 / sqrt(fact);
  _vector_mul_g(*v, fact, *v);
}
static void normalize_flt(suNg_vector_flt *v)
{
  float fact;
  _vector_prod_re_g(fact, *v, *v);
  fact = 1.0f / sqrtf(fact);
  _vector_mul_g(*v, fact, *v);
}
#endif

void project_to_suNg(suNg *u)
{
#ifdef GAUGE_SON
  double complex norm;
  _suNg_sqnorm(norm, *u);
  if (creal(norm) < 1.e-28)
    return;

  double *v1, *v2;
  int i, j, k;
  double z;
  for (i = 0; i < NG; ++i)
  {
    v2 = &u->c[i * NG];
    for (j = 0; j < i; ++j)
    {
      v1 = &u->c[j * NG];
      z = 0;
      for (k = 0; k < NG; ++k)
      {
        z += v1[k] * v2[k];
      } /*_vector_prod_re_g */
      for (k = 0; k < NG; ++k)
      {
        v2[k] -= z * v1[k];
      } /*_vector_project_g */
    }
    normalize(v2);
  }
#else
#ifdef WITH_QUATERNIONS
  double norm;
  _suNg_sqnorm(norm, *u);
  if (norm < 1.e-28)
    return;

  _suNg_sqnorm(norm, *u);
  norm = sqrt(0.5 * norm);
  norm = 1. / norm;
  _suNg_mul(*u, norm, *u);

#else
  double complex norm;
  _suNg_sqnorm(norm, *u);
  if (creal(norm) < 1.e-28)
    return;

  int i, j;
  suNg_vector *v1, *v2;
  double complex z;

  v1 = (suNg_vector *)(u);
  v2 = v1 + 1;
  normalize(v1);
  for (i = 1; i < NG; ++i)
  {
    for (j = i; j > 0; --j)
    {
      _vector_prod_g(z, *v1, *v2);
      _vector_project_g(*v2, z, *v1);
      ++v1;
    }
    normalize(v2);
    ++v2;
    v1 = (suNg_vector *)(u);
  }

  det_Cmplx_Ng(&norm, u);
  norm = cpow(norm, -1. / NG);
  _suNg_mul_assign(*u, norm);

#endif
#endif
}

void project_to_suNg_flt(suNg_flt *u)
{

#ifdef GAUGE_SON
  float norm;
  _suNg_sqnorm(norm, *u);
  if (norm < 1.e-10)
    return;

  float *v1, *v2;
  int i, j, k;
  float z;
  for (i = 0; i < NG; ++i)
  {
    v2 = &u->c[i * NG];
    for (j = 0; j < i; ++j)
    {
      v1 = &u->c[j * NG];
      z = 0;
      for (k = 0; k < NG; ++k)
      {
        z += v1[k] * v2[k];
      } /*_vector_prod_re_g */
      for (k = 0; k < NG; ++k)
      {
        v2[k] -= z * v1[k];
      } /*_vector_project_g */
    }
    normalize_flt(v2);
  }
#else
#ifdef WITH_QUATERNIONS
  float norm;
  _suNg_sqnorm(norm, *u);
  if (norm < 1.e-10)
    return;
  _suNg_sqnorm(norm, *u);
  norm = sqrtf(0.5f * norm);
  norm = 1.f / norm;
  _suNg_mul(*u, norm, *u);

#else
  float norm;
  double complex dnorm;
  suNg ud;
  _suNg_sqnorm(norm, *u);
  if (norm < 1.e-10)
    return;

  int i, j;
  suNg_vector_flt *v1, *v2;
  float complex z;

  v1 = (suNg_vector_flt *)(u);
  v2 = v1 + 1;

  normalize_flt(v1);
  for (i = 1; i < NG; ++i)
  {
    for (j = i; j > 0; --j)
    {
      _vector_prod_g(z, *v1, *v2);
      _vector_project_g(*v2, z, *v1);
      ++v1;
    }
    normalize_flt(v2);
    ++v2;
    v1 = (suNg_vector_flt *)(u);
  }
  for (i = 0; i < (NG * NG); ++i)
    ud.c[i] = (double complex)(u->c[i]);

  det_Cmplx_Ng(&dnorm, &ud);
  dnorm = cpow(dnorm, -1. / NG);
  _suNg_mul_assign(*u, dnorm);

#endif
#endif
}

#ifndef GAUGE_SON
void project_cooling_to_suNg(suNg *g_out, suNg *g_in, int cooling)
{
#ifdef WITH_QUATERNIONS
  error(1, 1, "project_cooling_to_suNg " __FILE__, "not implemented with quaternions");
#else
  suNg Ug[3];
  suNg tmp[2];
  int k, l;
  int j, i, ncool;
  double c[NG];
  double complex f[2];
  double norm;

  Ug[0] = *g_in;

  for (j = 0; j < NG; j++)
  {
    c[j] = 0.0;

    for (i = 0; i < NG; i++)
    {
      _complex_0(f[1]);

      for (k = 0; k < j; k++)
      {
        _complex_0(f[0]);

        for (l = 0; l < NG; l++)
        {
          _complex_mul_star_assign(f[0], (Ug[0]).c[l * NG + j], (Ug[1]).c[l * NG + k]);
        }
        _complex_mulcr_assign(f[1], c[k], (Ug[1]).c[i * NG + k], f[0]);
      }

      _complex_sub(Ug[1].c[i * NG + j], Ug[0].c[i * NG + j], f[1]);
      _complex_mul_star_assign_re(c[j], Ug[1].c[i * NG + j], Ug[1].c[i * NG + j]);
    }

    c[j] = 1.0 / c[j];
  }

  for (i = 0; i < NG; i++)
  {
    norm = 0.0;
    for (j = 0; j < NG; j++)
    {
      _complex_mul_star_assign_re(norm, Ug[1].c[i + NG * j], Ug[1].c[i + NG * j]);
    }

    for (j = 0; j < NG; j++)
    {
      _complex_mulr(Ug[1].c[i + NG * j], 1.0 / sqrt(norm), Ug[1].c[i + NG * j]);
    }
  }

  _suNg_dagger(Ug[2], *g_in);

  for (ncool = 0; ncool < cooling; ncool++)
  {

    _suNg_times_suNg(Ug[0], Ug[2], Ug[1]);

    for (i = 0; i < NG; i++)
    {
      for (j = i + 1; j < NG; j++)
      {

        _complex_add_star(f[0], Ug[0].c[i + NG * i], Ug[0].c[j + NG * j]);
        _complex_sub_star(f[1], Ug[0].c[j + NG * i], Ug[0].c[i + NG * j]);

        norm = 1.0 / sqrt(_complex_prod_re(f[0], f[0]) + _complex_prod_re(f[1], f[1]));

        _complex_mulr(f[0], norm, f[0]);
        _complex_mulr(f[1], norm, f[1]);

        _suNg_unit(tmp[0]);

        _complex_star(tmp[0].c[i + NG * i], f[0]);
        _complex_star(tmp[0].c[i + NG * j], f[1]);
        tmp[0].c[j + NG * j] = f[0];
        _complex_minus(tmp[0].c[j + NG * i], f[1]);

        _suNg_times_suNg(tmp[1], Ug[1], tmp[0]);
        Ug[1] = tmp[1];

        _suNg_times_suNg(tmp[1], Ug[0], tmp[0]);
        Ug[0] = tmp[1];
      }
    }
  }

  *g_out = Ug[1];
#endif
}
#endif

#ifdef GAUGE_SON

int project_to_suNg_real(suNg *out, suNg *in)
{
  suNg hm, om, tmp;
  double eigval[NG];
  double det;
  int i, j;
  _suNg_times_suNg_dagger(hm, *in, *in);
  diag_hmat(&hm, eigval);
  for (i = 0; i < NG; ++i)
  {
    eigval[i] = 1.0 / sqrt(eigval[i]);
  }
  for (i = 0; i < NG; ++i)
  {
    for (j = 0; j < NG; ++j)
    {
      tmp.c[i * NG + j] = eigval[j] * hm.c[i * NG + j];
    }
  }

  _suNg_times_suNg_dagger(om, tmp, hm);
  _suNg_times_suNg(tmp, om, *in);
  *out = tmp;
  // Fix the determinant
  det_Cmplx_Ng(&det, &tmp);
  /*  if (fabs(det)<1-1e-7 || fabs(det)>1+1e-7){
      lprintf("suNg_utils",10,"Error in project project_to_suNg_real: determinant not +/-1. It is %1.8g\n",det);
    }*/

  for (i = 0; i < NG; ++i)
  {
    out->c[i] *= 1. / det;
  }

  tmp = *out;
  det_Cmplx_Ng(&det, &tmp);
  if (det < 1 - 1e-7 || det > 1 + 1e-7)
  {
    lprintf("suNg_utils", 10, "Error in project project_to_suNg_real: determinant not +/-1. It is %1.8g.", det);
    return 0;
  }
  return 1;
}
#endif

void covariant_project_to_suNg(suNg *u)
{
  int i, j, k;
  suNg tmp, tmp1;
  double eval[NG];
  double complex norm;
  double complex evec[NG * NG];

  det_Cmplx_Ng(&norm, u);
  norm = cpow(norm, -1. / NG);
  _suNg_mul_assign(*u, norm);

  _suNg_dagger_times_suNg(tmp, *u, *u);

#ifdef WITH_QUATERNIONS
  /*With quaternions the only hermitian matrix that can be reqpresented is proportional to the idetity*/
  eval[0] = u->c[0];
  eval[1] = u->c[0];
  evec[0] = evec[3] = 1.0;
  evec[1] = evec[2] = 0.0;
#else
  jacobi2(NG, tmp.c, eval, evec);
#endif

  for (i = 0; i < NG; i++)
    eval[i] = 1 / sqrt(eval[i]);

  for (i = 0; i < NG; i++)
    for (j = 0; j < NG; j++)
      tmp1.c[NG * i + j] = eval[i] * conj(evec[NG * j + i]);

  for (i = 0; i < NG; i++)
    for (j = 0; j < NG; j++)
    {
      tmp.c[NG * i + j] = 0.;
      for (k = 0; k < NG; k++)
        tmp.c[NG * i + j] += evec[NG * i + k] * tmp1.c[NG * k + j];
    }

  tmp1 = *u;

  _suNg_times_suNg(*u, tmp1, tmp);
}
