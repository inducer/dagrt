* Incorporate timestepper states into language
    - specify states and transitions between them
* Code generation
    - Py code generator (see [utilities](https://github.com/inducer/pytools/blob/master/pytools/py_codegen.py))
    - Fortran code generator
    - Matlab code generator?
* Code generation of readable code
    - insert DAG and other information into comments
    - use structured control flow when possible
* Check if code is well-formed
    - Disallow side effects
* Implicit
* Incorporate IMEX
* Strang splitting transform
* Step matrix finder
* Fix ERK efficiency/reused subexpressions
    - implement CSE in code generator