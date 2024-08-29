/**
 * @file avr_plaquette.h
 * @brief Plaquette evaluation functions
 */

#ifndef AVR_PLAQUETTE_H
#define AVR_PLAQUETTE_H

#include "hr_complex.h"
#include "Core/spinor_field.h"

#ifdef __cplusplus
extern "C" {
#endif

// Local functions
double plaq(suNg_field *gauge, int ix, int mu, int nu);
void cplaq(hr_complex *ret, suNg_field *gauge, int ix, int mu, int nu);
void clover_F(suNg_algebra_vector *F, suNg_field *V, int ix, int mu, int nu);
double local_plaq(suNg_field *gauge, int ix);

// Observables
extern double (*avr_plaquette)(suNg_field *gauge);
extern void (*avr_plaquette_time)(suNg_field *gauge, double *plaqt, double *plaqs);
extern void (*full_plaquette)(suNg_field *gauge);
extern void (*local_plaquette)(suNg_field *gauge, scalar_field *s);
extern double (*E)(suNg_field *V);
extern void (*E_T)(double *E, suNg_field *V);
extern double (*Esym)(suNg_field *V);
extern void (*Esym_T)(double *E, suNg_field *V);
extern double (*topo)(suNg_field *V);

// Mostly necessary for testing
double avr_plaquette_gpu(suNg_field *u);
double avr_plaquette_cpu(suNg_field *u);
void avr_plaquette_time_cpu(suNg_field *u, double *plaqt, double *plaqs);
void avr_plaquette_time_gpu(suNg_field *u, double *plaqt, double *plaqs);
void full_plaquette_gpu(suNg_field *u);
void full_plaquette_cpu(suNg_field *u);

// Workspace functions
void avr_ts_plaquette(void);
void cplaq_wrk(hr_complex *ret, int ix, int mu, int nu);
hr_complex avr_plaquette_wrk(void);

#ifdef __cplusplus
}
#endif
#endif
