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
from pytools import UniqueNameGenerator


def wrap_line_base(line, level=0, width=80, indentation="    ",
                   pad_func=lambda string, amount: string,
                   lex_func=None):
    """
    The input is a line of code at the given indentation level. Return the list
    of lines that results from wrapping the line to the given width. Lines
    subsequent to the first line in the returned list are padded with extra
    indentation. The initial indentation level is not included in the input or
    output lines.

    The `pad_func` argument is a function that adds line continuations. The
    `lex_func` argument returns the list of tokens in the line.
    """
    if lex_func is None:
        lex_func = functools.partial(shlex.split, posix=False)

    tokens = lex_func(line)
    resulting_lines = []
    at_line_start = True
    indentation_len = len(level * indentation)
    current_line = ""
    padding_width = width - indentation_len
    for index, word in enumerate(tokens):
        has_next_word = index < len(tokens) - 1
        word_len = len(word)
        if not at_line_start:
            next_len = indentation_len + len(current_line) + 1 + word_len
            if next_len < width or (not has_next_word and next_len == width):
                # The word goes on the same line.
                current_line += " " + word
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


def remove_redundant_blank_lines(lines):
    def is_blank(line):
        return not line.strip()

    pending_blanks = []
    at_start = True

    result = []
    for line in lines:
        if is_blank(line):
            if not pending_blanks:
                pending_blanks.append(line)

        else:
            if not at_start:
                result.extend(pending_blanks)

            pending_blanks = []
            at_start = False

            result.append(line)

    return result


from string import ascii_letters, digits


_ident_chars = set("_" + ascii_letters + digits)


def make_identifier_from_name(name, default_identifier="dagrt_var"):
    result = "".join([c if c in _ident_chars else "_" for c in name])
    result = result.lstrip("_")
    if not result:
        result = default_identifier
    return result


class _KeyTranslatingUniqueNameGeneratorWrapper:

    def __init__(self, generator, translate):
        self._generator = generator
        self._translate = translate

    def add_name(self, name):
        return self._generator.add_name(name)

    def __call__(self, key):
        return self._generator(self._translate(key))


class KeyToUniqueNameMap:
    """Maps keys to unique names that are created on-the-fly.

    Before a unique name is created, a base name is first created. The base name
    consists of a forced prefix followed by a string that results from
    translating the key with a supplied function. The mapped value is then
    created from the base name.
    """

    def __init__(self, start=None, forced_prefix="",
                 key_translate_func=make_identifier_from_name,
                 name_generator=None):
        if start is None:
            start = {}
        self._dict = dict(start)

        if name_generator is None:
            name_generator = UniqueNameGenerator(forced_prefix=forced_prefix)
        else:
            if forced_prefix:
                raise TypeError("passing 'forced_prefix' is not allowed when "
                        "passing a pre-existing name generator")

        for existing_name in start.values():
            if existing_name.startswith(name_generator.forced_prefix):
                name_generator.add_name(existing_name)

        self._generator = _KeyTranslatingUniqueNameGeneratorWrapper(name_generator,
            key_translate_func)

    def get_or_make_name_for_key(self, key, prefix=None):
        try:
            return self._dict[key]
        except KeyError:
            seed = key
            if prefix is not None:
                seed = prefix+seed
            new_name = self._generator(seed)
            self._dict[key] = new_name
            return new_name

    def get_mapped_identifier_without_key(self, name):
        return self._generator(name)

    def __iter__(self):
        return iter(self._dict.keys())


def remove_common_indentation(text):
    """
    :arg text: a multi-line string
    :returns: a list of lines
    """

    if text is None:
        return text

    if not text.startswith("\n"):
        raise ValueError("expected newline as first character")

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
                raise ValueError("inconsistent indentation")

    return [line[base_indent:] for line in lines]
