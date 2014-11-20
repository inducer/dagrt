#! /bin/sh

set -e
set -x

python test_codegen_fortran.py \
  'test_rk_codegen(ODE23TimeStepper(use_high_order=True))' \
  > RKMethod.f90
gfortran -g -otestrk RKMethod.f90 test_integrator.f90
./testrk
