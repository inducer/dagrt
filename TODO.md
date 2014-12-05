* Implicit
* Incorporate timestepper states into language
    - specify states and transitions between them
* Code generation
    - Fortran code generator
    - Matlab code generator?
* Code generation of readable code
    - insert equations as comments
* Check if code is well-formed
    - Disallow side effects
* Incorporate IMEX
* Strang splitting transform
* Step matrix finder
* Fix ERK efficiency/reused subexpressions

* Single writer ideas
    - warn if a value is not defined yet
    - check for circular dependencies
    - copy_fence instruction for introducing temporaries and copies
    - nice visualization
