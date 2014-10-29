"""Random usefulness"""

__copyright__ = "Copyright (C) 2014 Matt Wala"

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

import functools
import shlex


def wrap_line_base(line, level=0, width=80, indentation='    ',
                   pad_func=lambda string, amount: string,
                   lex_func=functools.partial(shlex.split, posix=False)):
    """
    The input is a line of code at the given indentation level. Return the list
    of lines that results from wrapping the line to the given width. Lines
    subsequent to the first line in the returned list are padded with extra
    indentation. The initial indentation level is not included in the input or
    output lines.

    The `pad_func` argument is a function that adds line continuations. The
    `lex_func` argument returns the list of tokens in the line.
    """
    tokens = lex_func(line)
    resulting_lines = []
    at_line_start = True
    indentation_len = len(level * indentation)
    current_line = ''
    padding_width = width - indentation_len
    for index, word in enumerate(tokens):
        has_next_word = index < len(tokens) - 1
        word_len = len(word)
        if not at_line_start:
            next_len = indentation_len + len(current_line) + 1 + word_len
            if next_len < width or (not has_next_word and next_len == width):
                # The word goes on the same line.
                current_line += ' ' + word
            else:
                # The word goes on the next line.
                resulting_lines.append(pad_func(current_line, padding_width))
                at_line_start = True
                current_line = indentation
        if at_line_start:
            current_line += word
            at_line_start = False
    resulting_lines.append(current_line)
    return resulting_lines


def exec_in_new_namespace(code):
    namespace = {}
    exec(code, namespace)

    # pudb debugging support
    namespace["_MODULE_SOURCE_CODE"] = code

    return namespace


def chop_common_indentation(text):
    if text is None:
        return []

    if not text.startswith("\n"):
        raise ValueError("expected newline as first character "
                "in preamble")

    lines = text.split("\n")
    while lines[0].strip() == "":
        lines.pop(0)
    while lines[-1].strip() == "":
        lines.pop(-1)

    if lines:
        base_indent = 0
        while lines[0][base_indent] in " \t":
            base_indent += 1

        for line in lines[1:]:
            if line[:base_indent].strip():
                raise ValueError("inconsistent indentation in preamble")

    return [line[base_indent:] for line in lines]


def remove_redundant_blank_lines(lines):
    def is_blank(l):
        return not l.strip()

    pending_blanks = []
    at_start = True

    result = []
    for l in lines:
        if is_blank(l):
            if not pending_blanks:
                pending_blanks.append(l)

        else:
            if not at_start:
                result.extend(pending_blanks)

            pending_blanks = []
            at_start = False

            result.append(l)

    return result
