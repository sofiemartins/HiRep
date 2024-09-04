@page input_file Input File Configuration
[TOC]
## Lattice Geometry

Configure the size of your lattice by setting the variables

```
GLB_T = 8
GLB_X = 8
GLB_Y = 8
GLB_Z = 8
```

## Parallelization

Configure the decomposition into local lattices by determining the MPI decomposition. For example, the smallest local lattice is `3x3x3x3`, so an allowed configuration would be

```
GLB_T = 6
GLB_X = 6
GLB_Y = 6
GLB_Z = 6
NP_T = 2
NP_X = 2
NP_Y = 2
NP_Z = 2
```

using NP\_T*NP\_X*NP\_Y*NP\_Z=16 processes.

HiRep also allows decomposition into unevenly sized local lattices, meaning not all processes have the same local lattice size. For example, the following decomposition is allowed

```
GLB_T = 10
GLB_X = 8
GLB_Y = 8
GLB_Z = 8
NP_T = 3
NP_X = 2
NP_Y = 2
NP_Z = 2
```

cutting the lattice into local lattices of either `3x4x4x4` or `4x4x4x4` using 24 processes.

## Random number generator

This software uses RANLUX @cite Luscher:1993dy random numbers. You can configure this random number generator.

### Level

```
rlx_level = 1
```

configures the level of the RANLUX random number generator. These levels refer to the number of random numbers to be discarded in between two random numbers used for the simulation. In general the random number generation is not a bottleneck in the simulations, so choose the necessary quality here. A higher level indicates a higher quality of the random numbers in exchange for more necessary computation. However, `rlx_level=1` is usually sufficient for any application within the scope of this library.

### Seed

```
rlx_seed = 60718
```

The seed is only used when the random number generator is not initialized from a saved random number state. Please be very careful to consider the following notes.

1. It is very important that if you stop and restart a simulation, for example, with the HMC, you do not restart from the same seed; otherwise, unwanted and unphysical correlations will be introduced into your results. 
2. <span style="color:red">If you are using MPI, be aware that you are not using only a single seed but a seed range, where the first element is 60718, and the last element is 60718 + the number of processes x number of threads.</span> This means that if, for example, you have a parallelized simulation with 16 processes, stop and restart from a seed, and change your seed to 60719, your results will be correlated. You will have to set it to at least 60734.
3. While you can restart from a saved state, the saved state will only be valid for a specific threading and MPI layout. If you move to more or less processes and/or more or less threads, you must restart from a new seed.

### Random number state

```
rlx_state = rlx_state
```

This defines the file where the state will be saved. For reproducibility, it is also possible to store the random number states alongside the configurations. The default choice here is no, which means restoring the exact random number state will not be possible after a particular configuration. Note that the file above will continuously be updated/overwritten, while the states for reproducibility will be stored in the same directory as the configurations.

```
rlx_store = 1 // Store random number state alongside each configuration for reproducibility (default 0=no, 1=yes)
```

### Restarting

```
rlx_start = new
```

This option uses the seed to initialize a range of random number generators, one for each process. These states will then be written to file to restart from the latest state with the given MPI layout.

```
rlx_start = continue
```

This reads the random number state given in `rlx_state` and overwrites this with a new state when the configuration is written out. The recommended workflow for best reproducibility is to set `rlx_store=1` and then copy the state stored alongside the configuration to the file in the working directory given in `rlx_state`. 

## Logger

You may increase the logger level for debugging purposes (typical next levels would be 20, 50, or 100). However, in standard production settings, the default is sufficient.

```
//Logger levels (default = -1)
log:default = -1
log:inverter = -1
log:forcestat = -1
```

## Fermion twisting

This option is only available if you compiled the corresponding direction with `BC_\<DIR\>_THETA`. Then, you can set a global fermion phase in this direction by setting

```
theta_\<DIR\> = 0.
```

where 0. has to be replaced by the desired phase.

## HMC variables

```
tlen = 1.0
csw = 1.1329500
```

`tlen` corresponds to the length of the trajectory, and `csw` is the Sheikholeslami-Wohlert coefficient, which is only available if the code has been compiled with clover improvement, i.e., either with `WITH_CLOVER` or `WITH_EXPCLOVER`.

```
N_REP = 1
```

Number of replicas to be generated. With MPI, these are parallelized trivially.

```
run name = run1
```

The run name will be used to name the configurations written out.

```
save freq = 1
```

How often configurations should be saved. Putting `1` here implies that all configurations will be saved (every 1st trajectory) and putting `20` means every 20th trajectory will be saved.

```
meas freq = 1
```

If measurements are selected, this is the frequency with which they are done. For example, I could save only every 20th trajectory but select `2` here to measure the plaquette value and polyakov loops after every second trajectory. 

```
conf dir = cnfg
```

Specify here the directory in the current working directory where the configurations will be written out. The code does not create the directory and will fail if it does not exist, so you must create it before starting the simulation.

```
gauge start = random
```

Starting value for the gauge configuration. Possible values are `random`=hot start, `unit`=cold start, a file name of a file that is located in the directory given in `conf dir`=start up from a saved configuration. Note that you have to give the path of this file relative to the `conf dir`. For example, if the start-up configuration is called `run1_n0` and is located in `cnfg`, you put here `run1_n0` and **not** `cnfg/run1_n0`. 
  
```
last conf = +1
```

This option lets you specify the number of trajectories you want to generate. Note that there is a difference between `+1` and `1`. The "+" in front of `last conf` specifies the number of additional trajectories to be generated after the chosen startup configuration. I.e., if the startup configuration is trajectory number 5 and `last conf = 6`, then one additional trajectory will be generated, while if `last conf = +6`, then six additional trajectories will be generated (i.e., the last configuration generated will be number 11).

## Online measurements

### Eigenvalue tracking

```
eva:make = false
```
Set this to true to track eigenvalues for a selected number of standard operators at the end of the trajectories, given the frequency of measurements.

```
eva:nevt = 5
```

Search space dimension in searching for eigenvalues. This is limited by memory because fields need to be allocated for this, so if the application runs out of memory, try to reduce this number.

```
eva:nev = 1
```

Number of accurate eigenvalues that should be found.

```
eva:kmax = 50
eva:maxiter = 100
eva:omega1 = 1.e-3
eva:omega2 = 1.e-3
```

Parameters of the eigenvalue search algorithm: `eva:kmax` describes the maximum degree of the polynomial, `eva:maxiter` the maximum number of sub iterations, `eva:omega1` is the absolute precision, and `eva:omega2` the relative precision.

```
eva:mass = 1.0
```

Input mass for the eigenvalue calculations.

### Connected contributions to the correlation functions

```
mes:make = false
```

Set this to true to measure connected contributions to the correlation functions for a selected number of standard operators at the end of the trajectories, given the frequency of measurements.

```
mes:mass = -0.60
```

Input masses for the measurements. 

```
mes:precision = 1.e-24
```

Squared relative inverter precision. Here, the theoretical highest precision is 1.e-30, corresponding to 1.e-15 computer precision. The higher this precision, the more expensive this calculation is. Often, 1.e-16 is sufficient.

```
mes:nhits = 3
```

Number of sources used in the calculation. After the stochastic average, the simulation output for the connected contributions towards the correlation function will be done. So, the output will contain a single value for the correlation for each time slice and configuration.

```
mes:csw = 1.0
```

Value of the Sheikholeslami-Wohlert coefficient used during measurements.

### Polyakov loops

```
poly:make = false
```

Set this to `true` if you want the Polyakov loops measured at the end of trajectories, given the measurement frequency. 

## Integrators

The HiRep code uses a multilevel integrator; each integrator level must be specified in the input file.

```
    integrator {
        level = 0
        type = o2mn
        steps = 10
    }
```

|Variable|Description                                             |
|:-------|:-------------------------------------------------------|
|`level` |unique integrator level (level 0 is the outermost level)|
|`type`  |integrator type (see below)                             |
|`steps` |number of integration steps                             |

The table below shows the different integrators implemented in the HiRep code.
The last column in the table shows how many times the next integrator level is called in each iteration of the given integrator.

|Type  |Description                                |Next level calls|
:------|:------------------------------------------|:---------------|
|`lf`  |leap-frog integrator                       |1               |
|`o2mn`|2nd order minimal norm (omelyan) integrator|2               |
|`o4mn`|4th order minimal norm (omelyan) integrator|5               |

## Plaquette gauge

This gauge monomial is the standard Wilson plaquette action.

\f{equation}{ S = -\frac{\beta}{N}\sum_{x,\mu>\nu} \textrm{Re}~\textrm{tr}(U_\mu(x)U_\nu(x+\hat{\mu})U_\mu^\dagger(x+\hat{\nu})U_\nu^\dagger(x)) \f}

The following example shows how to specify a gauge monomial in the input file.

```
    monomial {
        id = 0
        type = gauge
        beta = 2.0
        level = 0
    }
```

|Variable|Description                                           |
|:-------|:-----------------------------------------------------|
|`id`    |unique monomial id                                    |
|`type`  |monomial type                                         |
|`beta`  |bare coupling for the gauge field                     |
|`level` |integrator level where the monomial force is evaluated|

## Lüscher-Weisz gauge

This gauge monomial is the Lüscher-Weisz (tree-level Symanzik) gauge action, including the \f$1\times1\f$ plaquettes \f$P_{\mu\nu}\f$ and the \f$1\times2\f$ rectangular loops \f$R_{\mu\nu}\f$.
The two coefficients below are related through \f$c_0+8c_1=1\f$ to ensure the correct continuum limit.

\f{equation}{ S = -\frac{\beta}{N}\sum_{x,\mu>\nu} c_0\textrm{Re}~\textrm{tr}[P_{\mu\nu}(x)] + c_1\textrm{Re}~\textrm{tr}[R_{\mu\nu}(x)+R_{\nu\mu}(x)] \f}

Specify a gauge monomial in the input file as in the following example:

```
    monomial {
        id = 0
        type = lw_gauge
        c0 = 1.666667
        beta = 2.0
        level = 0
    }
```

|Variable|Description                                           |
|:-------|:-----------------------------------------------------|
|`id`    |unique monomial id                                    |
|`type`  |monomial type                                         |
|`beta`  |bare coupling for the gauge field                     |
|`c0`    |coefficient in front of the plaquette term            |
|`level` |integrator level where the monomial force is evaluated|

## HMC Parameters

The HMC monomial is the standard term for simulating two mass degenerate fermions.

\f{equation}{ S = \phi^\dagger(D^\dagger D)^{-1}\phi\,, \f}

corresponding to the following input file configurations with example parameters:

```
    monomial {
        id = 1
        type = hmc
        mass = -0.750
        mt_prec = 1e-14
        force_prec = 1e-14
        mre_past = 5
        level = 1
    }
```

|Variable    |Description                                                |
|:-----------|:----------------------------------------------------------|
|`id`        |unique monomial id                                         |
|`type`      |monomial type                                              |
|`mass`      |bare fermion mass                                          |
|`mt_prec`   |inverter precision used in the Metropolis test             |
|`force_prec`|inverter precision used when calculating the force         |
|`mre_past`  |number of past solutions used in the chronological inverter|
|`level`     |integrator level where the monomial force is evaluated     |

## Twisted Mass

In this monomial the twisted mass is added before the Dirac operator has been even/odd preconditioned. Specify as follows:

```
    monomial {
        id = 1
        type = tm
        mass = -0.750
        mu = 0.1
        mt_prec = 1e-14
        force_prec = 1e-14
        mre_past = 5
        level = 1
    }
```

|Variable    |Description                                                |
|:-----------|:----------------------------------------------------------|
|`id`        |unique monomial id                                         |
|`type`      |monomial type                                              |
|`mass`      |bare fermion mass                                          |
|`mu`        |bare twisted mass                                          |
|`mt_prec`   |inverter precision used in the Metropolis test             |
|`force_prec`|inverter precision used when calculating the force         |
|`mre_past`  |number of past solutions used in the chronological inverter|
|`level`     |integrator level where the monomial force is evaluated     |

## Twisted Mass Alternative

In this monomial the twisted mass is added after the Dirac operator has been even-odd preconditioned.

```
    monomial {
        id = 1
        type = tm_alt
        mass = -0.750
        mu = 0.1
        mt_prec = 1e-14
        force_prec = 1e-14
        mre_past = 5
        level = 1
    }
```

|Variable    |Description                                                |
|:-----------|:----------------------------------------------------------|
|`id`        |unique monomial id                                         |
|`type`      |monomial type                                              |
|`mass`      |bare fermion mass                                          |
|`mu`        |bare twisted mass                                          |
|`mt_prec`   |inverter precision used in the Metropolis test             |
|`force_prec`|inverter precision used when calculating the force         |
|`mre_past`  |number of past solutions used in the chronological inverter|
|`level`     |integrator level where the monomial force is evaluated     |

## Hasenbusch

The Hasenbusch term is a mass preconditioned term used in connection with an HMC monomial.

\f{equation}{ S = \phi^\dagger\left(\frac{D^\dagger D}{(D+\Delta m)^\dagger (D+\Delta m)}\right)\phi \f}

```
    monomial {
        id = 1
        type = hasenbusch
        mass = -0.750
        dm = 0.1
        mt_prec = 1e-14
        force_prec = 1e-14
        mre_past = 2
        level = 0
    }
```

|Variable    |Description                                                |
|:-----------|:----------------------------------------------------------|
|`id`        |unique monomial id                                         |
|`type`      |monomial type                                              |
|`mass`      |bare fermion mass                                          |
|`dm`        |shift in the bare mass                                     |
|`mt_prec`   |inverter precision used in the Metropolis test             |
|`force_prec`|inverter precision used when calculating the force         |
|`mre_past`  |number of past solutions used in the chronological inverter|
|`level`     |integrator level where the monomial force is evaluated     |

## TM Hasenbusch

To include a Hasenbusch monomial with even-odd preconditioned twisted mass, adjust starting from the following template parameters


```
    monomial {
        id = 1
        type = hasenbusch_tm
        mass = -0.750
        mu = 0
        dmu = 0.1
        mt_prec = 1e-14
        force_prec = 1e-14
        mre_past = 2
        level = 0
    }
```

|Variable    |Description                                                |
|:-----------|:----------------------------------------------------------|
|`id`        |unique monomial id                                         |
|`type`      |monomial type                                              |
|`mass`      |bare fermion mass                                          |
|`mu`        |twisted mass                                               |
|`dmu`       |shift in the twisted mass                                  |
|`mt_prec`   |inverter precision used in the Metropolis test             |
|`force_prec`|inverter precision used when calculating the force         |
|`mre_past`  |number of past solutions used in the chronological inverter|
|`level`     |integrator level where the monomial force is evaluated     |

## TM Hasenbusch Alternative

For a twisted even-odd preconditioned operator, use the type `hasenbusch_tm_alt`.

```
    monomial {
        id = 1
        type = hasenbusch_tm_alt
        mass = -0.750
        mu = 0
        dmu = 0.1
        mt_prec = 1e-14
        force_prec = 1e-14
        mre_past = 2
        level = 0
    }
```

|Variable    |Description                                                |
|:-----------|:----------------------------------------------------------|
|`id`        |unique monomial id                                         |
|`type`      |monomial type                                              |
|`mass`      |bare fermion mass                                          |
|`mu`        |bare twisted mass                                          |
|`dmu`       |shift in the twisted mass                                  |
|`mt_prec`   |inverter precision used in the Metropolis test             |
|`force_prec`|inverter precision used when calculating the force         |
|`mre_past`  |number of past solutions used in the chronological inverter|
|`level`     |integrator level where the monomial force is evaluated     |

## RHMC

The RHMC monomial uses a rational approximation to simulate an odd number of mass degenerate fermions.

\f{equation}{ S = \phi^\dagger(D^\dagger D)^{-n/d}\phi \f}

Include this in the input file using the type `rhmc`. One further needs to specify numerator and denominator fractions in the rational approximation.  

```
    monomial {
        id = 1
        type = rhmc
        mass = -0.750
        n = 1
        d = 2
        mt_prec = 1e-14
        md_prec = 1e-14
        force_prec = 1e-14
        level = 0
    }
```

|Variable    |Description                                           |
|:-----------|:-----------------------------------------------------|
|`id`        |unique monomial id                                    |
|`type`      |monomial type                                         |
|`mass`      |bare fermion mass                                     |
|`n`         |fraction numerator                                    |
|`d`         |fraction denominator                                  |
|`mt_prec`   |inverter precision used in the Metropolis test        |
|`md_prec`   |precision of the rational approximation               |
|`force_prec`|inverter precision used when calculating the force    |
|`level`     |integrator level where the monomial force is evaluated|

## Chronological Inverter

When using the chronological inverter, the force precision should be \f$10^{-14}\f$ or better to ensure reversibility in the algorithm. Further, masses given in monomials should include the mass shift.

## Connected Contributions

```
mes:csw = 1.0
mes:precision = 1.e-14
```

Set Sheikholeslami-Wohlert coefficient and squared relative inverter precision. 

```
mes:nhits_2pt = 5
mes:mass = -0.45
```

Number of stochastic sources and input mass used in the connected contributions to the correlation function.

```
mes:meas_mixed = 1
```

Enable this (0=disabled, 1=enabled) if mixed channels should be measured, not just standard channels.

```
mes:momentum = 0
```

Maximum component of momenta. The default is 0. Some sources only allow this choice.

```
mes:def_semwall = 0
mes:def_point = 1
mes:def_gfwall = 0
mes:ext_semwall = 0
mes:ext_point = 0
```

Here, one can select the type of source used to measure the connected contributions. For each of the choices above that are enabled (disabled=0, enabled=1), the correlation function will be calculated and printed to the output. For the connected contributions, the stochastic average will be correctly evaluated automatically. Choices are

1. def_semwall: Default Spin-Explicit-Method Wall sources
2. def_point: Default point sources
3. def_gfwall: Default Gauge-fixed Wall sources
4. ext_semwall: Extended Spin-Explicit-Method Wall sources
5. ext_point: Extended point sources

```
mes:def_baryon = 0
```

Enable baryon measurements.

```
mes:dirichlet_dt = 2
mes:dirichlet_semwall = 0
mes:dirichlet_point = 0
mes:dirichlet_gfwall = 0
```

Sources with Dirichlet boundary conditions. Here, the options are 

1. dirichlet_semwall: Spin-Explicit-Method Wall sources with Dirichlet boundary conditions
2. dirichlet_point: Point sources with Dirichlet boundary conditions
3. dirichlet_gfwall: Gauge-fixed Wall sources with Dirichlet boundary conditions

Set `mes:dirichlet_dt` to the distance to the boundary.

```
mes:degree_hopping = 0
mes:nhits_hopping = 5
```

Settings for the hopping parameter expansion. Setting `mes:degree_hopping=0` disables the evaluation of correlation functions using the hopping parameter expansion otherwise use a positive integer as the order of the expansion. `mes:nhits_hopping` denotes the number of sources used in the stochastic average.

```
ff:on = false
```

Include four-fermion interactions.

```
mes:configlist = list_conf.txt
```

A list of paths where configurations are stored relative to the current working directory. This means that, for example, if `conf dir=cnfg`, this directory is included in the path, for example, `cnfg/run1_n100`. 

<span style="color:red">**Any settings for disconnected contributions in the folder `Spectrum` have no guarantee of working and are therefore not documented here. Please refer to the next section**</span>

## Disconnected Contributions

The binary in `Disconnected` is the only working code to evaluate disconnected contributions to correlation functions. In addition to the standard parameters to define the random numbers, lattice size and parallelization one has to set the following options:

```
disc:mass = -0.6
```

Is the input quark mass.

```
disc:precision = 1e-20
```

This is the squared relative inverter precision, i.e., this corresponds to a relative precision of 1e-10, and the maximum setting is 1e-30, which corresponds to 1e-15 relative precision (double precision).

```
disc:nhits = 2
```

The number of sources used for the calculation. This code will then produce one-point operator measurements for each time slice, channel, and source (plus additional measurements depending on the type of the source). The correlation function needs to be manually evaluated, taking the appropriate stochastic average and, if applicable, subtracting the vacuum expectation value. 

```
disc:source_type = 0
```

Type of source. Available types are 

* 0: Pure volume sources
* 1: Gauge fixed wall sources
* 2: Volume sources with time and spin dilution
* 3: Volume sources with time, spin, and color dilution
* 4: Volume sources with time, spin, color, and even-odd dilution
* 6: Volume source with spin, color, and even-odd dilution 

See more documentation on the sources in section `Analysis`.

```
disc:configlist = list_conf.txt
```

A list of paths where configurations are stored relative to the current working directory. This means that, for example, if `conf dir=cnfg`, this directory is included in the path, for example, `cnfg/run1_n100`. 

```
disc:n_mom = 1
```

Maximum component of the momentum, default: 1.

## Wilson Flow

```
WF:integrator = 2
```

Wilson flow integrator used. Options are

1. 0: Euler integration
2. 1: 3rd-order Runge-Kutta integration
3. 2: Adaptive 3rd-order Runge-Kutta integration

```
WF:tmax = 0.2
```

Maximal evolution time in the Wilson flow evolution.

```
WF:nmeas = 1
```

Number of measurements. Putting one here measures once at `WF:tmax`, in this example at 0.2 flow time. Putting 100 here will measure after each step of 2e-3.

```
WF:eps = 0.002
WF:delta= 0.00001
```

Adaptive epsilon and delta parameters for the (adaptive) integrator.

```
WF:configlist = list_confs.txt
```