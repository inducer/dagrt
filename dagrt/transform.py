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

def insert_empty_intermediate_state(dag, state_name, after_state):
    new_states = dag.phases.copy()

    if state_name in new_states:
        raise ValueError("state '%s' already exists"
                % state_name)

    from dagrt.language import DAGCode, ExecutionPhase
    new_states[state_name] = ExecutionPhase(
            next_phase=new_states[after_state].next_phase)
    new_states[after_state] = new_states[after_state].copy(
            next_phase=state_name)

    return DAGCode(dag.instructions, new_states, dag.initial_phase)

# }}}


# {{{ fuse_two_states

def fuse_two_states(state_name, state1, state2):
    from dagrt.language import ExecutionPhase
    if state1 is not None and state2 is not None:
        if state1.next_phase != state2.next_phase:
            raise ValueError("DAGs don't agree on default "
                    "state transition out of state '%s'"
                    % state_name)

        from pymbolic.imperative.transform import disambiguate_and_fuse
        new_instructions, _, old_2_id_to_new_2_id = disambiguate_and_fuse(
                state1.instructions, state2.instructions)

        return ExecutionPhase(
                next_phase=state1.next_phase,
                depends_on=frozenset(state1.depends_on) | frozenset(
                    old_2_id_to_new_2_id.get(id2, id2)
                    for id2 in state2.depends_on),
                instructions=new_instructions
                )

    elif state1 is not None:
        return state1
    elif state2 is not None:
        return state2
    else:
        raise ValueError("both states are None")

# }}}


# {{{ fuse_two_dags

def fuse_two_dags(dag1, dag2, state_correspondences=None,
        should_disambiguate_name=None):
    from dagrt.language import DAGCode
    new_states = {}
    for state_name in frozenset(dag1.phases) | frozenset(dag2.phases):
        state1 = dag1.phases.get(state_name)
        state2 = dag2.phases.get(state_name)

        new_states[state_name] = fuse_two_states(state_name, state1, state2)

    if dag1.initial_phase != dag2.initial_phase:
        raise ValueError("DAGs don't agree on initial state")

    return DAGCode(new_states, dag1.initial_phase)

# }}}


# vim: foldmethod=marker
