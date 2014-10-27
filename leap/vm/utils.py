"""Various useful functions for working with the timestepper description
language."""

from __future__ import division, with_statement

__copyright__ = """
Copyright (C) 2014 Matt Wala
Copyright (C) 2014 Andreas Kloeckner
"""

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


def get_variables(expr):
    """Returns the set of names of variables used in the expression."""
    from leap.vm.expression import variable_mapper
    return frozenset(dep.name for dep in variable_mapper(expr))


def is_state_variable(var):
    """Check if the given name corresponds to a state variable."""
    if var == '<t>' or var == '<dt>':
        return True
    elif var.startswith('<state>') or var.startswith('<p>'):
        return True
    else:
        return False


def get_unique_name(prefix, *name_sets):
    """Return a name that begins with prefix and is not found in any of the
    name sets."""

    def is_in_name_sets(var):
        for name_set in name_sets:
            if var in name_set:
                return True
        return False

    prefix = str(prefix)
    if not is_in_name_sets(prefix):
        return prefix

    suffix = 0
    while is_in_name_sets(prefix + str(suffix)):
        suffix += 1
    return prefix + str(suffix)


def show_dot_graph(dot):
    """Displays the given DOT format string in the web browser."""
    from tempfile import mkdtemp
    temp_dir = mkdtemp(prefix="tmp_leap_dot")

    dot_file_name = "leap.dot"

    from os.path import join
    with open(join(temp_dir, dot_file_name), "w") as dotf:
        dotf.write(dot)

    svg_file_name = "leap.svg"
    from subprocess import check_call
    check_call(["dot", "-Tsvg", "-o", svg_file_name, dot_file_name],
                cwd=temp_dir)

    full_svg_file_name = join(temp_dir, svg_file_name)

    from webbrowser import open as browser_open
    browser_open("file://" + full_svg_file_name)


class TODO(NotImplementedError):
    pass


def resolve_args(arg_names, default_dict, arg_dict):
    """Resolve positional and keyword arguments to a single argument
    list.

    :arg arg_dict: a dictionary mapping numbers (for positional arguments)
        or identifiers (for keyword arguments) to values
    :arg default_dict: a dictionary mapping argument names to default
        values
    :arg arg_names: names of the positional arguments
    """

    arg_dict = arg_dict.copy()
    args = []
    for i, name in enumerate(arg_names):
        if i in arg_dict:
            args.append(arg_dict.pop(i))
            if name in arg_dict:
                raise TypeError("argument '%d' specified both "
                        "positionally and by keyword" % arg_names[i])
        elif name in arg_dict:
            args.append(arg_dict.pop(name))
        else:
            if name in default_dict:
                args.append(default_dict[name])
            else:
                raise TypeError("argument '%s' not specified" % arg_names[i])

    if arg_dict:
        raise TypeError("leftover arguments after argument resolution: "
                + ", ".join(arg_dict))

    return args
