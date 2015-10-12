* Code generation
    - Matlab code generator?
* Check if code is well-formed
* Strang splitting transform
* Step matrix finder
* Fix ERK efficiency/reused subexpressions

* Solver ideas:
    - Remove AssignSolved, replace with AssignExpression
* Method ideas:
    - Simplify interface. Deprecate set_up(). Don't pass two separate contexts.
