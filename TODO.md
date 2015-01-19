* Implicit
* Code generation
    - Matlab code generator?
* Check if code is well-formed
    - Disallow side effects
* Strang splitting transform
* Step matrix finder
* Fix ERK efficiency/reused subexpressions

* Solver ideas:
    - Remove AssignSolved, replace with AssignExpression
* Method ideas:
    - Simplify interface. Deprecate set_up(). Don't pass two separate contexts.
* AB ideas
    - Implement transitions, add an RK state and a Primary state
