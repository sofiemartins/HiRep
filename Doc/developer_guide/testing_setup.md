@page testing_setup Testing Setup
## Unit Tests

### Nocompile

The test suite has to be able to test a number of different setups generated by different choices in `Make/MkFlags`. Some of these choices, for example compiling with or without GPU, will also generate different namespaces. However, we still want to preserve the ability to test only for a single case of choosing the compilation flags.

The testing `Makefile`s have therefore been configured to only run certain tests for certain compilation flags, using a `NOCOMPILE = XXX` statement, where `XXX` is the corresponding compilation flag. If we, for example, want to write a test that works only if compiles without GPU acceleration, we can use this flag to configure the test this way in the preamble

```
/*******************************************************************************
*
* NOCOMPILE= WITH_GPU
*
* This test is only compiled if the WITH_GPU flag is inactive
*
*******************************************************************************/
```

Notice, that there is no space between `NOCOMPILE` and `=`. Conversely, we can configure tests that only test the GPU-version by negation

```
/*******************************************************************************
*
* NOCOMPILE= !WITH_GPU
*
* This tests only the GPU-version of HiRep
*
*******************************************************************************/
```

If we want to make sure that multiple flags are active, we can connect them over `&&`.

```
/*******************************************************************************
*
* NOCOMPILE= !WITH_GPU && !WITH_MPI
*
* This tests only the Multi-GPU compiled version of HiRep
*
*******************************************************************************/
```

If the test does test multiple, but not all possible setups, we can write them down using multiple lines.

```
/*******************************************************************************
*
* NOCOMPILE= WITH_GPU
* NOCOMPILE= WITH_MPI
*
* This test can be compiled either if MPI or GPU acceleration is disabled.
*
*******************************************************************************/
```

## Integration Tests

## Test Report Generation