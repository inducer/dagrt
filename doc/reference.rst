Reference
=========

.. module:: leap

Description Language
--------------------

.. automodule:: leap.vm.language

Instructions
~~~~~~~~~~~~
.. autoclass:: Instruction

Assignment Instructions
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: AssignRHS

.. autoclass:: AssignSolvedRHS

.. autoclass:: AssignExpression

.. autoclass:: AssignNorm

.. autoclass:: AssignDotProduct

State Instructions
^^^^^^^^^^^^^^^^^^

.. autoclass:: ReturnState

.. autoclass:: Raise

.. autoclass:: FailStep

.. autoclass:: If

Code Container
~~~~~~~~~~~~~~

.. autoclass:: TimeIntegratorCode

Visualization
~~~~~~~~~~~~~

.. autofunction:: get_dot_dependency_graph

.. autofunction:: show_dependency_graph

Methods
-------

.. automodule:: leap.method.rk

.. autoclass:: ODE23TimeStepper

.. autoclass:: ODE45TimeStepper

Execution
---------

:mod:`numpy`-based interpretation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: leap.vm.exec_numpy

.. autoclass:: NumpyInterpreter

.. _numpy-exec-events:

Events
^^^^^^

.. autoclass:: StateComputed

.. autoclass:: StepCompleted

.. autoclass:: StepFailed
