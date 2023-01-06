/*******************************************************************************
*
* NOCOMPILE= !WITH_GPU
*
* Gauge covariance of the Dirac operator on GPU
*
*******************************************************************************/

#include "libhr.h"

static double hmass = 0.1;
static suNg_field *g;

static void loc_D(spinor_field *out, spinor_field *in)
{
  Dphi(hmass, out, in);
}

static void random_g(void)
{
  _MASTER_FOR(&glattice, ix)
  {
    random_suNg(_FIELD_AT(g, ix));
  }
}

static void transform_u(void)
{
  _MASTER_FOR(&glattice, ix)
  {
    suNg v;
    for (int mu = 0; mu < 4; mu++)
    {
      int iy = iup(ix, mu);
      suNg *u = pu_gauge(ix, mu);
      _suNg_times_suNg_dagger(v, *u, *_FIELD_AT(g, iy));
      _suNg_times_suNg(*u, *_FIELD_AT(g, ix), v);
    }
  }

  start_gf_sendrecv(u_gauge);
  represent_gauge_field();
  smear_gauge_field();
}

static void transform_s(spinor_field *out, spinor_field *in)
{
  _MASTER_FOR(&glattice, ix)
  {
    suNf_spinor *s = _FIELD_AT(in, ix);
    suNf_spinor *r = _FIELD_AT(out, ix);
    suNf gfx;

    _group_represent2(&gfx, _FIELD_AT(g, ix));

    _suNf_multiply(r->c[0], gfx, s->c[0]);
    _suNf_multiply(r->c[1], gfx, s->c[1]);
    _suNf_multiply(r->c[2], gfx, s->c[2]);
    _suNf_multiply(r->c[3], gfx, s->c[3]);
  }
}

int main(int argc, char *argv[])
{

  int return_value = 0;
  double sig, tau;
  spinor_field *s0, *s1, *s2, *s3;

  // Setup process
  logger_map("DEBUG", "debug");
  setup_process(&argc, &argv);

  // Setup gauge fields
  setup_gauge_fields();
  g = alloc_gtransf(&glattice); /* allocate additional memory */
  lprintf("MAIN", 0, "Generating a random gauge field... ");
  fflush(stdout);
  random_u(u_gauge);
  start_gf_sendrecv(u_gauge);
  represent_gauge_field();
  copy_to_gpu_gfield_f(u_gauge_f);
  lprintf("MAIN", 0, "done.\n");

  // Generate random gauge transformation to apply
  lprintf("MAIN", 0, "Generating a random gauge transf... ");
  random_g();
  copy_to_gpu_gfield(g);
  lprintf("MAIN", 0, "done.\n");

  // Setup initial gauge fields
  s0 = alloc_spinor_field_f(4, &glattice);
  s1 = s0 + 1;
  s2 = s1 + 1;
  s3 = s2 + 1;
  spinor_field_zero_f(s0);
  gaussian_spinor_field(&(s0[0]));
  
  // Normalize s0 + Sanity check
  tau = 1. / sqrt(spinor_field_sqnorm_f_cpu(s0));
  spinor_field_mul_f_cpu(s0, tau, s0);
  sig = spinor_field_sqnorm_f_cpu(s0);
  lprintf("MAIN", 0, "Normalized norm = %.2e\n", sig);

  // Apply Gauge TF
  lprintf("MAIN", 0, "Gauge covariance of the Dirac operator:\n");
  copy_to_gpu_spinor_field_f(s0);
  loc_D(s1, s0);
  copy_from_gpu_spinor_field_f(s1);

  transform_s(s2, s1);
  transform_s(s3, s0);
  transform_u();

  copy_to_gpu_spinor_field_f(s2);
  copy_to_gpu_spinor_field_f(s3);
  copy_to_gpu_gfield_f(u_gauge_f);
  spinor_field_zero_f(s1);
  loc_D(s1, s3);
  spinor_field_mul_add_assign_f(s1, -1.0, s2);
  sig = spinor_field_sqnorm_f(s1);

  // Print test results
  lprintf("MAIN", 0, "Maximal normalized difference = %.2e\n", sqrt(sig));
  lprintf("MAIN", 0, "(should be around 1*10^(-15) or so)\n");

  if (sqrt(sig) > 10.e-14) 
  {
    lprintf("RESULT", 0, "FAILED \n");
    return_value = 1;
  } 
  else lprintf("RESULT", 0, "OK \n");

  // Free and return
  free_spinor_field_f(s0);
  free_gtransf(g);
  finalize_process();
  return return_value;
}