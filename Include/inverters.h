/***************************************************************************\
* Copyright (c) 2022, Claudio Pica                                          *   
* All rights reserved.                                                      * 
\***************************************************************************/

#ifndef INVERTERS_H
#define INVERTERS_H

#include "Inverters/precise_sums.h"
#ifdef __cplusplus
#include "Inverters/precise_sums_gpu.hpp"
#include "Inverters/linear_algebra_generic_gpu.hpp"
#else
#include "Inverters/linear_algebra_generic.h"
#endif

#include "Inverters/linear_algebra.h"
#include "Inverters/linear_solvers.h"
#include "Inverters/global_sum.h"
#include "Inverters/scalarfield_operations.h"
#include "Inverters/test_complex.h"

#endif
