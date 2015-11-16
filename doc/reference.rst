Reference
=========

.. module:: dagrt

Description language
--------------------

.. automodule:: dagrt.language

Code generation
---------------

Python
~~~~~~

.. automodule:: dagrt.codegen.python

.. autoclass:: CodeGenerator

Fortran
~~~~~~~

.. automodule::	dagrt.codegen.fortran

.. autoclass:: CodeGenerator

Function registry
~~~~~~~~~~~~~~~~~

The function registry is used by targets to register external
functions and customized function call code.

.. automodule:: dagrt.function_registry
   :members:

Transformations
~~~~~~~~~~~~~~~

.. automodule:: dagrt.codegen.transform
   :members:

Utilities
~~~~~~~~~

.. automodule:: dagrt.codegen.utils
   :members:

.. automodule:: dagrt.utils
   :members:


:mod:`numpy`-based interpretation
---------------------------------

.. automodule:: dagrt.exec_numpy
   :members:
