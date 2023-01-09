This repository contains the HiRep simulation code.

# Getting started

## Dependencies

* A C99 compiler (GCC, clang, icc). OpenMP can be used if supported by the compiler.  
* If MPI is needed, an MPI implementation, i.e. OpenMPI or MPICH for MPI support. Use a CUDA-aware MPI implementation for multi-GPU support.
* If GPU acceleration if needed, CUDA 11.x and nvcc compiler to make use of CUDA GPU acceleration.
* Perl 5.x for compilation.
* [ninja build](https://ninja-build.org/) for compilation.


## Compilation

### Clone the directory

```
git clone https://github.com/claudiopica/HiRep
```

Make sure the build command `Make/nj` and `ninja` are in your `PATH`. 

### Adjust compilation options 
Adjust the file `Make/MkFlags` to set the desired options.
The option file can be generated by using the `Make/Utils/write_mkflags.pl` tools.
Use:
```
write_mkflags.pl -h
```
for a list of available options. The most important ones include:

* Number of colors (NG)
```
NG=3
```

* Gauge group SU(NG) or SO(NG)
```
GAUGE_GROUP = GAUGE_SUN
#GAUGE_GROUP = GAUGE_SON
```

* Representation of fermion fields
```
REPR = REPR_FUNDAMENTAL
#REPR = REPR_SYMMETRIC
#REPR = REPR_ANTISYMMETRIC
#REPR = REPR_ADJOINT
```

* Lattice boundary Conditions

Comment out the line here, when you want to establish certain boundary conditions into the respective direction.
```
#Available choices of boundary conditions:
#T => PERIODIC, ANTIPERIODIC, OPEN, THETA
#X => PERIODIC, ANTIPERIODIC, THETA
#Y => PERIODIC, ANTIPERIODIC, THETA
#Z => PERIODIC, ANTIPERIODIC, THETA
MACRO += -DBC_T_ANTIPERIODIC
MACRO += -DBC_X_PERIODIC
MACRO += -DBC_Y_PERIODIC
MACRO += -DBC_Z_PERIODIC
```

* MACRO options

You can select a number of features via the `MACRO` variable. The most important ones are:

Specify, whether you want to compile with MPI by using 

```
#MACRO += -DWITH_MPI
```

For compilation with GPU acceleration for CUDA GPUs use:

```
MACRO += -DWITH_GPU
```

* Compiler options

You can set your choice of C, C++, MPI and CUDA compiler and their options by using the variables:
```
CC = gcc
MPICC = mpicc
NVCC = nvcc
CXX = g++
LDFLAGS = -Wall -O3
GPUFLAGS = -arch=sm_80 
INCLUDE = 
```

For example, to use the Intel compiler and Intel's MPI implementation, and no CUDA, one could use:

```
CC = icc
MPICC = mpiicc
LDFLAGS = -O3
INCLUDE = 
```

### Compile the code
From the root folder just type:
```
nj
```
(this is a tool in the `Make/` folder: make sure it is in your path!)
The above will compile the `libhr.a` library and all the available executable in the HiRep distribution, including executable for dnamical fermions `hmc` and pure gauge `suN` simulations and all the applicable tests.
If you wish to compile only one of the executable, e.g. `suN`, just change to the corresponding directory, e.g. `PureGauge`, and execute the `nj` command from there.

All build artefacts, except the final executables, are located in the `build` folder at the root directory of the distribution.


## Run

### Adjust input file
As example we will use the `hmc` program which can be found in the ```HMC``` directory (to create the executable type `nj` in that directory). 
The `hmc` program will run the generation of lattice configurations with dynamical fermions by using a hybrid Monte Carlo algorithm. The program uses a number of parameters which needs to be specified in an input file, see ```HMC/input_file``` for an example. 
Input parameters are divided in different sections, such as: global lattice size, number of MPI processes per direction, random number generator, run control variables, definition of the lattice action to use for the run, etc.
For example, for basic run control variables, one can have a look at the section ```Run control variables```.

```
run name = run1
save freq = 1
meas freq = 1
conf dir = cnfg
gauge start = random 
last conf = +1
```

The "+" in front of ```last conf``` specifies the number of additional trajectories to be generated after the chosen startup configuration. I.e. if the startup configuration is trajectory number 5 and ```last conf = 6``` then one additional trajectory will be generated, while if ```last conf = +6``` then six additional trajectories will be generated (i.e. the last configuration generated will be number 11).

### Execute Binary

When not using MPI, simply run:

```
$ ./hmc -i input_file
```

where ```hmc``` is the binary generated from ```hmc.c```. If you are using openmp, remeber to set `OMP_NUM_THREADS` and other relevant environment variables to the desired value.

For the MPI version, run

```
$ mpirun -np <number of MPI processes> ./hmc -i input_file
```

The GPU version of the code uses 1 GPU per MPI process.

The output file is written only by the MPI process rank 0, by default in a file called `out_0` in the current directory. A different name for the output file can be set by using the `-o` option.

For debug purposes it is sometimes useful to have output files from all MPI processes. This can be enabled with the compilation option: `MACRO += -DLOG_ALLPIDS`.


# Documentation

* [Github pages](https://claudiopica.github.io/HiRep/)


# How To Cite

Luigi Del Debbio, Agostino Patella and Claudio Pica. "Higher representations on the lattice: Numerical simulations, SU(2) with adjoint fermions". In: _Phys. Rev. D_ **81** (9 2010) p. 094503. DOI: [10.1103/PhysRevD.81.094503](https://doi.org/10.1103/PhysRevD.81.094503). URL: [https://link.aps.org/doi/10.1103/PhysRevD.81.094503](https://link.aps.org/doi/10.1103/PhysRevD.81.094503)


![https://github.com/claudiopica/HiRep/actions?workflow=no-mpi-dev](https://github.com/claudiopica/HiRep/workflows/no-mpi-dev/badge.svg)
![https://github.com/claudiopica/HiRep/actions?workflow=mpi-dev](https://github.com/claudiopica/HiRep/workflows/mpi-dev/badge.svg)
