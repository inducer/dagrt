"""Dagrt root module"""

__copyright__ = "Copyright (C) 2014 Andreas Kloeckner"

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


def run_script_from_commandline():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("script", metavar="SCRIPT.PY")
    parser.add_argument("args", metavar="ARG", nargs="*")
    args = parser.parse_args()

    from os.path import abspath, dirname
    scriptdir = dirname(abspath(args.script))

    import sys
    sys.argv[1:] = args.args
    sys.path.append(scriptdir)

    with open(args.script) as s:
        script_contents = s.read()

    namespace = {"__name__": "__main__"}
    exec(compile(script_contents, args.script, "exec"), namespace)
