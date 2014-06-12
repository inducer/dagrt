The leap code generator
=======================

Stages of code generation
-------------------------

1. The DAG is partitioned to blocks of straight-line code.
2. The control-flow graph is generated from the partitioned DAG.
3. Optimizations run on the control-flow graph.
4. The optimized control-flow graph is converted to a control tree.
5. The control tree is passed to language specific code generator.

Stage 1: Partitioning the DAG
-----------------------------

See: `dag2ir.py`

To prepare the DAG for code generations, it is subjected to simplifications:

1. *Pruning of unnecessary instructions:* The DAG contains sub-DAGs for each
   major stage of the timestepper description. Only the instructions relevant to
   the particular stage we're generating code for are kept.

2. *Entry and exit marker insertion:* Dummy "Entry" and "Exit" instructions get inserted
   into the DAG. These instructions serve as markers for the entry and exit
   points of the eventual function.

3. *Pruning of unnecessary unconditional dependencies:* Redundant unconditional
   dependencies are removed (as an optimization) via transitive reduction of the
   DAG's unconditional edges.

4. *Partitioning into blocks:* The DAG is partitioned into "blocks" of
   straight-line code, with each instruction inside the block having a single
   predecessor and successor.

Stage 2: Control-flow graph generation
--------------------------------------

See: `dag2ir.py`

The basic procedure is to build a control-flow subgraph corresponding to a block
of code and its dependencies, starting with the block containing the "Exit"
instruction. The following algorithm gives a description of what is done for
code without "If" statements:

    Gen_Subgraph(Instruction Block without "If" statements):
	   1. Recursively generate the subgraphs for each dependency block of the
          instruction block.
	   2. Combine the dependency subgraphs in sequential order.
	   3. Generate a basic block containing the instructions in the instruction
          block and append the new basic block at the end of the generated
          subgraph.

"If" statements need to be handled in a different manner. To execute an "If"
statement, the condition is evaluated before one of the conditional dependency
edges is taken (thus, the first difficulty with the above algorithm is that the
basic blocks corresponding to the conditional edges need to go after the
generated conditional statement). Furthermore, the semantics of the DAG is such
that after an "If" instruction is executed, each conditional dependency on the
appropriate branch is executed exactly once. The main challenge in control flow
graph generation with "If" statements is tracking of dependencies through
conditional execution paths so that each dependency is indeed executed once.

This is solved by introducing a boolean flag variable for each instruction block
which records if the instruction block has been executed. Before an instruction
block is executed, a check is performed on the flag to determine if the block
needs to be executed (as an optimization, such checks are elided if on all
incoming paths the block has not been executed). At the end of the execution of
a block of instructions, the associated block flag is set to true.

Stage 3: Optimization
---------------------

See: `optimization.py`

The following optimizations are run on the control-flow graph:

1. Aggressive dead code elimination removes dead code, including dead
   flags introduced by the control flow graph generation step.

2. Control-flow graph simplification, merges straight-line sequences of basic
   blocks, coalesces chains of trivial jumps, and removes unreachable basic
   blocks.

Stage 4: Control tree generation
--------------------------------

See: `ir2structured_ir.py`

The control tree resembles an abstract syntax tree. Each node of the tree
represents an identified control structure in the DAG. The following control
structures are supported:

* Simple basic block
* Blocks: a sequence of control structures
* If-Then
* If-Then-Else
* Unstructured interval: any other unrecognized control structure

The structural extractor creates a control tree that can be passed to the high
level code generator.

Coding conventions
==================

Optimization conventions
------------------------

* The optimization passes are implemented in `optimization.py.`

* Optimizations can directly modify the code that is passed to them.

* Optimizations are called through the `__call__` method. If any modifications
  are done to the input, return True. Return False otherwise.

* Parameters to optimizations should be passed through constructors.

* Optimizations should not retain state across calls.

Analysis conventions
--------------------

* Analysis passes are in `analysis.py`.

* Analyses should not modify the input object in any way.

* Analyses are called by constructing an analysis object. Slow analyses should
  have a separate construction method. (Hence they cannot be re-used for
  different objects.)
