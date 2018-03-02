"""Fusion and other user-facing code transforms"""

__copyright__ = "Copyright (C) 2016 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


# {{{ insert_empty_intermediate_state

def insert_empty_intermediate_state(dag, phase_name, after_phase):
    new_phases = dag.phases.copy()

    if phase_name in new_phases:
        raise ValueError("phase '%s' already exists"
                % phase_name)

    from dagrt.language import DAGCode, ExecutionPhase
    new_phases[phase_name] = ExecutionPhase(
            next_phase=new_phases[after_phase].next_phase)
    new_phases[after_phase] = new_phases[after_phase].copy(
            next_phase=phase_name)

    return DAGCode(dag.instructions, new_phases, dag.initial_phase)

# }}}


# {{{ fuse_two_phases

def fuse_two_phases(phase_name, phase1, phase2):
    from dagrt.language import ExecutionPhase
    if phase1 is not None and phase2 is not None:
        if phase1.next_phase != phase2.next_phase:
            raise ValueError("DAGs don't agree on default "
                    "phase transition out of phase '%s'"
                    % phase_name)

        from pymbolic.imperative.transform import disambiguate_and_fuse
        new_instructions, _, old_2_id_to_new_2_id = disambiguate_and_fuse(
                phase1.instructions, phase2.instructions)

        return ExecutionPhase(
                next_phase=phase1.next_phase,
                depends_on=frozenset(phase1.depends_on) | frozenset(
                    old_2_id_to_new_2_id.get(id2, id2)
                    for id2 in phase2.depends_on),
                instructions=new_instructions
                )

    elif phase1 is not None:
        return phase1
    elif phase2 is not None:
        return phase2
    else:
        raise ValueError("both phases are None")

# }}}


# {{{ fuse_two_dags

def fuse_two_dags(dag1, dag2, phase_correspondences=None,
        should_disambiguate_name=None):
    from dagrt.language import DAGCode
    new_phases = {}
    for phase_name in frozenset(dag1.phases) | frozenset(dag2.phases):
        phase1 = dag1.phases.get(phase_name)
        phase2 = dag2.phases.get(phase_name)

        new_phases[phase_name] = fuse_two_phases(phase_name, phase1, phase2)

    if dag1.initial_phase != dag2.initial_phase:
        raise ValueError("DAGs don't agree on initial phase")

    return DAGCode(new_phases, dag1.initial_phase)

# }}}


# vim: foldmethod=marker
