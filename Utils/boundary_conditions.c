 /***************************************************************************\
* Copyright (c) 2008, Claudio Pica                                          *   
* All rights reserved.                                                      * 
\***************************************************************************/

#include <math.h>

#include "global.h"
#include "utils.h"
#include "suN.h"
#include "communications.h"


#include "observables.h"

#ifdef ROTATED_SF
#include "update.h"
extern rhmc_par _update_par; /* Update/update_rhmc.c */
#endif /* ROTATED_SF */

/***************************************************************************/
/* THETA BC */
/***************************************************************************/

void set_bc_theta_t(double theta)
{
#ifdef BC_T_THETA
    eitheta[0].re=cos(theta/(double)GLB_T);
    eitheta[0].im=sin(theta/(double)GLB_T);
#endif //BC_T_THETA
}

void set_bc_theta_x(double theta)
{
#ifdef BC_X_THETA
    eitheta[1].re=cos(theta/(double)GLB_X);
    eitheta[1].im=sin(theta/(double)GLB_X);
#endif //BC_X_THETA
}

void set_bc_theta_y(double theta)
{
#ifdef BC_Y_THETA
    eitheta[2].re=cos(theta/(double)GLB_Y);
    eitheta[2].im=sin(theta/(double)GLB_Y);
#endif //BC_Y_THETA
}

void set_bc_theta_z(double theta)
{
#ifdef BC_Z_THETA
    eitheta[3].re=cos(theta/(double)GLB_Z);
    eitheta[3].im=sin(theta/(double)GLB_Z);
#endif //BC_Z_THETA
}

/***************************************************************************/



void apply_bc(){
#if defined(ANTIPERIODIC_BC_T) && !defined(ROTATED_SF) && !defined(BASIC_SF)
  if(COORD[0]==0) {
    int index;
    int ix,iy,iz;
    suNf *u;
    for (ix=0;ix<X_EXT;++ix) for (iy=0;iy<Y_EXT;++iy) for (iz=0;iz<Z_EXT;++iz){
      index=ipt_ext(2*T_BORDER,ix,iy,iz);
      if(index!=-1) {
	u=pu_gauge_f(index,0);
	_suNf_minus(*u,*u);
      }
    }
  }
#elif defined(ROTATED_SF)
  if(COORD[0] == 0) {
    int index;
    int ix,iy,iz;
    suNf *u;
    for (ix=0;ix<X_EXT;++ix) for (iy=0;iy<Y_EXT;++iy) for (iz=0;iz<Z_EXT;++iz){
      index=ipt_ext(T_BORDER+1,ix,iy,iz);
      if(index!=-1) {
	u=pu_gauge_f(index,1);
	_suNf_mul(*u,_update_par.SF_ds,*u);
	u=pu_gauge_f(index,2);
	_suNf_mul(*u,_update_par.SF_ds,*u);
	u=pu_gauge_f(index,3);
	_suNf_mul(*u,_update_par.SF_ds,*u);
      }
    }
  }
  if(COORD[0] == NP_T-1) {
    int index;
    int ix,iy,iz;
    suNf *u;
    for (ix=0;ix<X_EXT;++ix) for (iy=0;iy<Y_EXT;++iy) for (iz=0;iz<Z_EXT;++iz){
      index=ipt_ext(T+T_BORDER-1,ix,iy,iz);
      if(index!=-1) {
	u=pu_gauge_f(index,1);
	_suNf_mul(*u,_update_par.SF_ds,*u);
	u=pu_gauge_f(index,2);
	_suNf_mul(*u,_update_par.SF_ds,*u);
	u=pu_gauge_f(index,3);
	_suNf_mul(*u,_update_par.SF_ds,*u);
      }
    }
  }
#endif

#ifdef ANTIPERIODIC_BC_X
  if(COORD[1]==0) {
    int index;
    int it,iy,iz;
    suNf *u;
    for (it=0;it<T_EXT;++it)
      for (iy=0;iy<Y_EXT;++iy)
	for (iz=0;iz<Z_EXT;++iz){
	  index=ipt_ext(it,2*X_BORDER,iy,iz);
	  if(index!=-1) {
	    suNf *u=pu_gauge_f(index,1);
	    _suNf_minus(*u,*u);
	  }
	}
  }
#endif
  
#ifdef ANTIPERIODIC_BC_Y
  if(COORD[2]==0) {
    int index;
    int ix,it,iz;
    suNf *u;
    for (it=0;it<T_EXT;++it)
      for (ix=0;ix<X_EXT;++ix)
	for (iz=0;iz<Z_EXT;++iz){
	  index=ipt_ext(it,ix,2*Y_BORDER,iz);
	  if(index!=-1) {
	    suNf *u=pu_gauge_f(index,2);
	    _suNf_minus(*u,*u);
	  }
	}
  }
#endif
  
#ifdef ANTIPERIODIC_BC_Z
  if(COORD[3]==0) {
    int index;
    int ix,iy,it;
    suNf *u;
    for (it=0;it<T_EXT;++it)
      for (ix=0;ix<X_EXT;++ix)
	for (iy=0;iy<Y_EXT;++iy){
	  index=ipt_ext(it,ix,iy,2*Z_BORDER);
	  if(index!=-1) {
	    suNf *u=pu_gauge_f(index,3);
	    _suNf_minus(*u,*u);
	  }
	}
  }
#endif

}


void apply_bc_flt(){
#if defined(ANTIPERIODIC_BC_T) && !defined(ROTATED_SF) && !defined(BASIC_SF)
  if(COORD[0]==0) {
    int index;
    int ix,iy,iz;
    suNf_flt *u;
    for (ix=0;ix<X_EXT;++ix) for (iy=0;iy<Y_EXT;++iy) for (iz=0;iz<Z_EXT;++iz){
      index=ipt_ext(2*T_BORDER,ix,iy,iz);
      if(index!=-1) {
	u=pu_gauge_f_flt(index,0);
	_suNf_minus(*u,*u);
      }
    }
  }
#elif defined(ROTATED_SF)
  if(COORD[0] == 0) {
    int index;
    int ix,iy,iz;
    suNf_flt *u;
    for (ix=0;ix<X_EXT;++ix) for (iy=0;iy<Y_EXT;++iy) for (iz=0;iz<Z_EXT;++iz){
      index=ipt_ext(T_BORDER+1,ix,iy,iz);
      if(index!=-1) {
	u=pu_gauge_f_flt(index,1);
	_suNf_mul(*u,((float)_update_par.SF_ds),*u);
	u=pu_gauge_f_flt(index,2);
	_suNf_mul(*u,((float)_update_par.SF_ds),*u);
	u=pu_gauge_f_flt(index,3);
	_suNf_mul(*u,((float)_update_par.SF_ds),*u);
      }
    }
  }
  if(COORD[0] == NP_T-1) {
    int index;
    int ix,iy,iz;
    suNf_flt *u;
    for (ix=0;ix<X_EXT;++ix) for (iy=0;iy<Y_EXT;++iy) for (iz=0;iz<Z_EXT;++iz){
      index=ipt_ext(T+T_BORDER-1,ix,iy,iz);
      if(index!=-1) {
	u=pu_gauge_f_flt(index,1);
	_suNf_mul(*u,((float)_update_par.SF_ds),*u);
	u=pu_gauge_f_flt(index,2);
	_suNf_mul(*u,((float)_update_par.SF_ds),*u);
	u=pu_gauge_f_flt(index,3);
	_suNf_mul(*u,((float)_update_par.SF_ds),*u);
      }
    }
  }
#endif
  
#ifdef ANTIPERIODIC_BC_X
  if(COORD[1]==0) {
    int index;
    int it,iy,iz;
    suNf_flt *u;
    for (it=0;it<T_EXT;++it) for (iy=0;iy<Y_EXT;++iy) for (iz=0;iz<Z_EXT;++iz){
      index=ipt_ext(it,2*X_BORDER,iy,iz);
      if(index!=-1) {
	u=pu_gauge_f_flt(index,1);
	_suNf_minus(*u,*u);
      }
    }
  }
#endif

#ifdef ANTIPERIODIC_BC_Y
  if(COORD[2]==0) {
    int index;
    int ix,it,iz;
    suNf_flt *u;
    for (it=0;it<T_EXT;++it) for (ix=0;ix<X_EXT;++ix) for (iz=0;iz<Z_EXT;++iz){
      index=ipt_ext(it,ix,2*Y_BORDER,iz);
      if(index!=-1) {
	u=pu_gauge_f_flt(index,2);
	_suNf_minus(*u,*u);
      }
    }
  }
#endif

#ifdef ANTIPERIODIC_BC_Z
  if(COORD[3]==0) {
    int index;
    int ix,iy,it;
    suNf_flt *u;
    for (it=0;it<T_EXT;++it) for (ix=0;ix<X_EXT;++ix) for (iy=0;iy<Y_EXT;++iy){
      index=ipt_ext(it,ix,iy,2*Z_BORDER);
      if(index!=-1) {
	u=pu_gauge_f_flt(index,3);
	_suNf_minus(*u,*u);
      }
    }
  }
#endif

}

#if defined(BASIC_SF) || defined(ROTATED_SF)

void SF_spinor_bcs(spinor_field *sp)
{
#ifdef BASIC_SF
  int it,ix,iy,iz,index;
  
  if(COORD[0] == 0) {
    for (ix=0;ix<X;++ix) for (iy=0;iy<Y;++iy) for (iz=0;iz<Z;++iz){
      for(it=0;it<=1;it++) {
	index=ipt(it,ix,iy,iz);
	_spinor_zero_f(*_FIELD_AT(sp,index));
      }
    }
  }
  if(COORD[0] == NP_T-1) {
    for (ix=0;ix<X;++ix) for (iy=0;iy<Y;++iy) for (iz=0;iz<Z;++iz){
      index=ipt(T-1,ix,iy,iz);
      _spinor_zero_f(*_FIELD_AT(sp,index));
    }
  }
#else 
  int ix,iy,iz,index;
  
  if(COORD[0] == 0) {
    for (ix=0;ix<X;++ix) for (iy=0;iy<Y;++iy) for (iz=0;iz<Z;++iz){
      index=ipt(0,ix,iy,iz);
      _spinor_zero_f(*_FIELD_AT(sp,index));
    }
  }
#endif
}

void SF_spinor_bcs_flt(spinor_field_flt *sp)
{
#ifdef BASIC_SF
 int it,ix,iy,iz,index;
  
  if(COORD[0] == 0) {
    for (ix=0;ix<X;++ix) for (iy=0;iy<Y;++iy) for (iz=0;iz<Z;++iz){
      for(it=0;it<=1;it++) {
	index=ipt(it,ix,iy,iz);
	_spinor_zero_f(*_FIELD_AT(sp,index));
      }
    }
  }
  if(COORD[0] == NP_T-1) {
    for (ix=0;ix<X;++ix) for (iy=0;iy<Y;++iy) for (iz=0;iz<Z;++iz){
      index=ipt(T-1,ix,iy,iz);
      _spinor_zero_f(*_FIELD_AT(sp,index));
    }
  }

#else 
  int ix,iy,iz,index;
  
  if(COORD[0] == 0) {
    for (ix=0;ix<X;++ix) for (iy=0;iy<Y;++iy) for (iz=0;iz<Z;++iz){
      index=ipt(0,ix,iy,iz);
      _spinor_zero_f(*_FIELD_AT(sp,index));
    }
  }
#endif
}


double SF_test_spinor_bcs(spinor_field *sp) {
/*should be zero*/
	double pa=0., k;
	int ix,iy,iz,index;

	if(COORD[0] == 0) {
		for (ix=0;ix<X;++ix) for (iy=0;iy<Y;++iy) for (iz=0;iz<Z;++iz){
			index=ipt(0,ix,iy,iz);
			_spinor_prod_re_f(k,*_FIELD_AT(sp,index),*_FIELD_AT(sp,index)); pa+=k;
			_spinor_prod_im_f(k,*_FIELD_AT(sp,index),*_FIELD_AT(sp,index)); pa+=k;
			index=ipt(1,ix,iy,iz);
			_spinor_prod_re_f(k,*_FIELD_AT(sp,index),*_FIELD_AT(sp,index)); pa+=k;
			_spinor_prod_im_f(k,*_FIELD_AT(sp,index),*_FIELD_AT(sp,index)); pa+=k;
		}
	}
	if(COORD[0] == NP_T-1) {
		for (ix=0;ix<X;++ix) for (iy=0;iy<Y;++iy) for (iz=0;iz<Z;++iz){
			index=ipt(T-1,ix,iy,iz);
			_spinor_prod_re_f(k,*_FIELD_AT(sp,index),*_FIELD_AT(sp,index)); pa+=k;
			_spinor_prod_im_f(k,*_FIELD_AT(sp,index),*_FIELD_AT(sp,index)); pa+=k;
		}
	}
  global_sum(&pa, 1);
  return pa/(double)(GLB_X*GLB_Y*GLB_Z*(4*NF)*(3));
}

#endif /* BASIC_SF */





#if defined(BASIC_SF) || defined(ROTATED_SF)
/*We should test if the use of even/odd could give any performance benefit*/
void SF_force_bcs(suNg_av_field *force) {
  int ix,iy,iz,index;
  
  if(COORD[0] == 0) {
    for (ix=0;ix<X;++ix) for (iy=0;iy<Y;++iy) for (iz=0;iz<Z;++iz){
      index=ipt(0,ix,iy,iz);
      _algebra_vector_zero_g(*_4FIELD_AT(force,index,0));
      _algebra_vector_zero_g(*_4FIELD_AT(force,index,1));
      _algebra_vector_zero_g(*_4FIELD_AT(force,index,2));
      _algebra_vector_zero_g(*_4FIELD_AT(force,index,3));
      
      index=ipt(1,ix,iy,iz);
      _algebra_vector_zero_g(*_4FIELD_AT(force,index,1));
      _algebra_vector_zero_g(*_4FIELD_AT(force,index,2));
      _algebra_vector_zero_g(*_4FIELD_AT(force,index,3));
    }
  }
  if(COORD[0] == NP_T-1) {
    for (ix=0;ix<X;++ix) for (iy=0;iy<Y;++iy) for (iz=0;iz<Z;++iz){
	index=ipt(T-1,ix,iy,iz);
	_algebra_vector_zero_g(*_4FIELD_AT(force,index,0));
	_algebra_vector_zero_g(*_4FIELD_AT(force,index,1));
	_algebra_vector_zero_g(*_4FIELD_AT(force,index,2));
	_algebra_vector_zero_g(*_4FIELD_AT(force,index,3));
    }
  }
  
}




double SF_test_force_bcs(suNg_av_field *force) {
/*should be zero*/
  double pa=0., k;
  int ix,iy,iz,index;
  
  if(COORD[0] == 0) {
    for (ix=0;ix<X;++ix) for (iy=0;iy<Y;++iy) for (iz=0;iz<Z;++iz){
      index=ipt(0,ix,iy,iz);
      _algebra_vector_sqnorm_g(k,*_4FIELD_AT(force,index,0)); pa+=k;
      _algebra_vector_sqnorm_g(k,*_4FIELD_AT(force,index,1)); pa+=k;
      _algebra_vector_sqnorm_g(k,*_4FIELD_AT(force,index,2)); pa+=k;
      _algebra_vector_sqnorm_g(k,*_4FIELD_AT(force,index,3)); pa+=k;
      index=ipt(1,ix,iy,iz);
      _algebra_vector_sqnorm_g(k,*_4FIELD_AT(force,index,1)); pa+=k;
      _algebra_vector_sqnorm_g(k,*_4FIELD_AT(force,index,2)); pa+=k;
      _algebra_vector_sqnorm_g(k,*_4FIELD_AT(force,index,3)); pa+=k;
    }
  }
  if(COORD[0] == NP_T-1) {
    for (ix=0;ix<X;++ix) for (iy=0;iy<Y;++iy) for (iz=0;iz<Z;++iz){
      index=ipt(T-1,ix,iy,iz);
      _algebra_vector_sqnorm_g(k,*_4FIELD_AT(force,index,0)); pa+=k;
      _algebra_vector_sqnorm_g(k,*_4FIELD_AT(force,index,1)); pa+=k;
      _algebra_vector_sqnorm_g(k,*_4FIELD_AT(force,index,2)); pa+=k;
      _algebra_vector_sqnorm_g(k,*_4FIELD_AT(force,index,3)); pa+=k;
    }
  }
  global_sum(&pa, 1);
  return pa/(double)(GLB_X*GLB_Y*GLB_Z*(NG*NG-1)*(4+4+3));
}



#define PI 3.141592653589793238462643383279502884197
#define ST 1.414213562373095048801688724209698078570


#if NG==2

double SF_eta = PI/4.0;
double SF_phi0_dn[NG] = {0., 0.};
double SF_phi1_dn[NG] = {-1., 1.};
double SF_phi0_up[NG] = {-PI, PI};
double SF_phi1_up[NG] = {1., -1.};

#elif NG==3

double SF_eta = 0.;
double SF_phi0_dn[NG] = {-PI/3., 0., PI/3.};
double SF_phi1_dn[NG] = {1., -.5, -.5};
double SF_phi0_up[NG] = {-PI, PI/3., 2.*PI/3.};
double SF_phi1_up[NG] = {-1., .5, .5};

#elif NG==4

double SF_eta = 0.;
double SF_phi0_dn[NG] = {-ST*PI/4., ST*PI/4.-PI/2., PI/2.-ST*PI/4., ST*PI/4.};
double SF_phi1_dn[NG] = {-.5, -.5, .5, .5};
double SF_phi0_up[NG] = {-ST*PI/4.-PI/2., -PI+ST*PI/4., PI-ST*PI/4., PI/2.+ST*PI/4.};
double SF_phi1_up[NG] = {.5, .5, -.5, -.5};

#else

#error SF boundary conditions not defined at this NG

#endif


void SF_gauge_bcs(suNg_field *gf, int strength)
{
  int index;
  int ix, iy, iz;
  int k;
  suNg *u;
  
  error(gf==NULL,1,"SF_gauge_bcs [random_fields.c]",
	"Attempt to access unallocated memory space");   
  
  /*Boundary gauge fields*/
  suNg Bound0, BoundT;
  if(strength==1) {  /*SF bcs*/
    _suNg_zero(Bound0);
    for(k=0; k<NG; k++) {
      Bound0.c[(1+NG)*k].re = cos((SF_phi0_dn[k]+SF_phi1_dn[k]*SF_eta)/(GLB_T-2));
      Bound0.c[(1+NG)*k].im = sin((SF_phi0_dn[k]+SF_phi1_dn[k]*SF_eta)/(GLB_T-2));
    }
    _suNg_zero(BoundT);
    for(k=0; k<NG; k++) {
      BoundT.c[(1+NG)*k].re = cos((SF_phi0_up[k]+SF_phi1_up[k]*SF_eta)/(GLB_T-2));
      BoundT.c[(1+NG)*k].im = sin((SF_phi0_up[k]+SF_phi1_up[k]*SF_eta)/(GLB_T-2));
    }
  } else { /*UNIT bcs*/
    _suNg_unit(Bound0);
    _suNg_unit(BoundT);
  }	
  
  if(COORD[0] == 0) {
    for (ix=0;ix<X;++ix) for (iy=0;iy<Y;++iy) for (iz=0;iz<Z;++iz){
      index=ipt(0,ix,iy,iz);
      u=((gf->ptr)+coord_to_index(index,0));
      _suNg_unit(*u);
      u=((gf->ptr)+coord_to_index(index,1));
      _suNg_unit(*u);
      u=((gf->ptr)+coord_to_index(index,2));
      _suNg_unit(*u);
      u=((gf->ptr)+coord_to_index(index,3));
      _suNg_unit(*u);
      
      index=ipt(1,ix,iy,iz);
      u=((gf->ptr)+coord_to_index(index,1));
      *u = Bound0;
      u=((gf->ptr)+coord_to_index(index,2));
      *u = Bound0;
      u=((gf->ptr)+coord_to_index(index,3));
      *u = Bound0;
    }
  }
  if(COORD[0] == NP_T-1) {
    for (ix=0;ix<X;++ix) for (iy=0;iy<Y;++iy) for (iz=0;iz<Z;++iz){
      index=ipt(T-1,ix,iy,iz);
      u=((gf->ptr)+coord_to_index(index,0));
      _suNg_unit(*u);
      u=((gf->ptr)+coord_to_index(index,1));
      *u = BoundT;
      u=((gf->ptr)+coord_to_index(index,2));
      *u = BoundT;
      u=((gf->ptr)+coord_to_index(index,3));
      *u = BoundT;
    }
    
  }
  
  start_gf_sendrecv(gf);
}




double SF_test_gauge_bcs()
{
  /*calculates average of all plaquettes that should remain fixed for SF*/
  double pa=0.;
  int ix, iy, iz,index;
  
  start_gf_sendrecv(u_gauge);
  complete_gf_sendrecv(u_gauge);
  
  if(COORD[0] == 0) {
    for (ix=0;ix<X;++ix) for (iy=0;iy<Y;++iy) for (iz=0;iz<Z;++iz){
      index=ipt(0,ix,iy,iz);
      pa+=(double)(plaq(index,1,0));
      pa+=(double)(plaq(index,2,0));
      pa+=(double)(plaq(index,2,1));
      pa+=(double)(plaq(index,3,0));
      pa+=(double)(plaq(index,3,1));
      pa+=(double)(plaq(index,3,2));
      index=ipt(1,ix,iy,iz);
      pa+=(double)(plaq(index,2,1));
      pa+=(double)(plaq(index,3,1));
      pa+=(double)(plaq(index,3,2));
    }
  }
  if(COORD[0] == NP_T-1) {
    for (ix=0;ix<X;++ix) for (iy=0;iy<Y;++iy) for (iz=0;iz<Z;++iz){
      index=ipt(T-1,ix,iy,iz);
      pa+=(double)(plaq(index,1,0));
      pa+=(double)(plaq(index,2,0));
      pa+=(double)(plaq(index,3,0));
      pa+=(double)(plaq(index,2,1));
      pa+=(double)(plaq(index,3,1));
      pa+=(double)(plaq(index,3,2));
    }
  }
  global_sum(&pa, 1);
  return pa/(double)(GLB_X*GLB_Y*GLB_Z*NG*(6+6+3));
}

#endif //defined(BASIC_SF) || defined(ROTATED_SF)
