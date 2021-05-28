"""Dumping ground of miscellaneous helpfulness."""

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

import sys


class TODO(NotImplementedError):
    pass


def get_variables(expr, include_function_symbols=False):
    """Returns the set of names of variables used in the expression."""
    from dagrt.expression import ExtendedDependencyMapper
    args = {"include_subscripts": False,
            "include_lookups": False,
            "include_calls": "descend_args"
            if not include_function_symbols else False}
    variable_mapper = ExtendedDependencyMapper(**args)
    return frozenset(dep.name for dep in variable_mapper(expr))


# {{{ name wrangling

def is_state_variable(var):
    """Check if the given name corresponds to a state variable."""
    if var in ("<t>", "<dt>"):
        return True
    elif (var.startswith("<state>")
            or var.startswith("<p>")
            or var.startswith("<ret_time_id>")
            or var.startswith("<ret_time>")
            or var.startswith("<ret_state>")
          ):
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

# }}}


# {{{ resolve_args

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
                + ", ".join(str(i) for i in arg_dict))

    return tuple(args)

# }}}


# {{{ temporary directory

class TemporaryDirectory:
    """Create and return a temporary directory.  This has the same
    behavior as mkdtemp but can be used as a context manager.  For
    example:

        with TemporaryDirectory() as tmpdir:
            ...

    Upon exiting the context, the directory and everything contained
    in it are removed.
    """

    # Yanked from
    # https://hg.python.org/cpython/file/3.3/Lib/tempfile.py

    # Handle mkdtemp raising an exception
    name = None
    _closed = False

    def __init__(self, suffix="", prefix="tmp", dirname=None):
        from tempfile import mkdtemp
        self.name = mkdtemp(suffix, prefix, dirname)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.name!r}>"

    def __enter__(self):
        return self.name

    def cleanup(self, _warn=False):
        import warnings
        if self.name and not self._closed:
            from shutil import rmtree
            rmtree(self.name)
            self._closed = True
            if _warn and warnings.warn:
                warnings.warn(f"Implicitly cleaning up {self!r}")

    def __exit__(self, exc, value, tb):
        self.cleanup()

    def __del__(self):
        # Issue a ResourceWarning if implicit cleanup needed
        self.cleanup(_warn=True)

# }}}


# {{{ run_fortran

class DebuggerExit(Exception):
    pass


def run_fortran(sources, fortran_options=None, fortran_libraries=None, debug=False):
    if fortran_options is None:
        fortran_options = []
    if fortran_libraries is None:
        fortran_libraries = []

    from os.path import join

    with TemporaryDirectory() as tmpdir:
        source_names = []
        for name, contents in sources:
            source_names.append(name)

            with open(join(tmpdir, name), "w") as srcf:
                srcf.write(contents)

        import os
        from subprocess import check_call, Popen, PIPE
        check_call(
                [os.environ.get("FC", "gfortran"),
                    "-Wall",
                    "-Wno-unused-dummy-argument",
                    "-Wno-unused-variable",
                    "-Wno-maybe-uninitialized",
                    "-g", "-oruntest"]
                + fortran_options
                + list(source_names)
                + ["-l"+lib for lib in fortran_libraries],
                cwd=tmpdir)

        if debug:
            p = Popen(["gdb", "--args", join(tmpdir, "runtest")])
            p.wait()
            raise DebuggerExit
        else:
            p = Popen([join(tmpdir, "runtest")], stdout=PIPE, stderr=PIPE,
                    close_fds=True)
            stdout_data, stderr_data = p.communicate()

            if stdout_data:
                print("Fortran code said this on stdout: -------------------------",
                        file=sys.stderr)
                print(stdout_data.decode(), file=sys.stderr)
                print("-----------------------------------------------------------",
                        file=sys.stderr)

        if stderr_data:
            raise RuntimeError(
                    "Fortran code has non-empty stderr:\n"
                    + stderr_data.decode("ascii"))

        return p.returncode, stdout_data, stderr_data

# }}}

# vim: foldmethod=marker
