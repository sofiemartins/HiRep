/*******************************************************************************
*
* File eo_lexi.c
*
* Definition of the lattice geometry
*
*******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "geometry.h"
#include "global.h"
#include "safe_mod.h"

static int index_eo_lexi(int x0,int x1,int x2,int x3)
{
   int y0,y1,y2,y3;
   int ix;
   
   y0=safe_mod(x0,T);
   y1=safe_mod(x1,L);
   y2=safe_mod(x2,L);
   y3=safe_mod(x3,L);

   ix = (y3+L*(y2+L*(y1+L*y0)))/2;
   if((x0+x1+x2+x3)&1){
      ix+=VOLUME/2;
   }

   return ix;
      
}

void geometry_eo_lexi(void)
{
   int x0,x1,x2,x3,ix;

   for (x0=0;x0<T;x0++){
     for (x1=0;x1<L;x1++){
       for (x2=0;x2<L;x2++){
	 for (x3=0;x3<L;x3++){
	   ix=index_eo_lexi(x0,x1,x2,x3);
	   ipt[x0][x1][x2][x3]=ix;
	   
	   iup[ix][0]=index_eo_lexi(x0+1,x1,x2,x3);
	   idn[ix][0]=index_eo_lexi(x0-1,x1,x2,x3);
	   iup[ix][1]=index_eo_lexi(x0,x1+1,x2,x3);
	   idn[ix][1]=index_eo_lexi(x0,x1-1,x2,x3);
	   iup[ix][2]=index_eo_lexi(x0,x1,x2+1,x3);
	   idn[ix][2]=index_eo_lexi(x0,x1,x2-1,x3);
	   iup[ix][3]=index_eo_lexi(x0,x1,x2,x3+1);
	   idn[ix][3]=index_eo_lexi(x0,x1,x2,x3-1);
	   /* tslice[ix]=x0; */
	 }
       }
     }
   }
}

