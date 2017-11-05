"""Python built-in functions"""

__copyright__ = "Copyright (C) 2015 Matt Wala, Andreas Kloeckner"

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


def builtin_len(x):
    import numpy as np
    return np.size(x)


def builtin_isnan(x):
    import numpy as np
    return np.isnan(x)


def builtin_norm_1(x):
    import numpy as np
    if np.isscalar(x):
        return abs(x)
    return np.linalg.norm(x, 1)


def builtin_norm_2(x):
    import numpy as np
    if np.isscalar(x):
        return abs(x)
    return np.linalg.norm(x, 2)


def builtin_norm_inf(x):
    import numpy as np
    if np.isscalar(x):
        return abs(x)
    return np.linalg.norm(x, np.inf)


def builtin_dot_product(a, b):
    import numpy as np
    return np.vdot(a, b)


def builtin_array(n):
    import numpy as np
    if n != np.floor(n):
        raise ValueError("array() argument n is not an integer")
    n = int(n)

    return np.empty(n, dtype=np.float64)


def builtin_matmul(a, b, a_cols, b_cols):
    import numpy as np
    if a_cols != np.floor(a_cols):
        raise ValueError("matmul() argument a_cols is not an integer")
    if b_cols != np.floor(b_cols):
        raise ValueError("matmul() argument b_cols is not an integer")
    a_cols = int(a_cols)
    b_cols = int(b_cols)

    a_mat = a.reshape(-1, a_cols, order="F")
    b_mat = b.reshape(-1, b_cols, order="F")

    res_mat = a_mat.dot(b_mat)

    return res_mat.reshape(-1, order="F")


def builtin_transpose(a, a_cols):
    import numpy as np
    if a_cols != np.floor(a_cols):
        raise ValueError("transpose() argument a_cols is not an integer")
    a_cols = int(a_cols)

    a_mat = a.reshape(-1, a_cols, order="F")

    res_mat = np.transpose(a_mat)

    return res_mat.reshape(-1, order="F")


def builtin_linear_solve(a, b, a_cols, b_cols):
    import numpy as np
    if a_cols != np.floor(a_cols):
        raise ValueError("linear_solve() argument a_cols is not an integer")
    if b_cols != np.floor(b_cols):
        raise ValueError("linear_solve() argument b_cols is not an integer")
    a_cols = int(a_cols)
    b_cols = int(b_cols)

    a_mat = a.reshape(-1, a_cols, order="F")
    b_mat = b.reshape(-1, b_cols, order="F")

    import numpy.linalg as la
    res_mat = la.solve(a_mat, b_mat)

    return res_mat.reshape(-1, order="F")


def builtin_svd(a, a_cols):
    import numpy as np
    if a_cols != np.floor(a_cols):
        raise ValueError("linear_solve() argument a_cols is not an integer")
    a_cols = int(a_cols)

    a_mat = a.reshape(-1, a_cols, order="F")

    import numpy.linalg as la
    u, sigma, vt = la.svd(a_mat, full_matrices=0)

    return u.reshape(-1, order="F"), sigma, vt.reshape(-1, order="F")


def builtin_print(arg):
    print(arg)


builtins = {
        "<builtin>len": builtin_len,
        "<builtin>isnan": builtin_isnan,
        "<builtin>norm_1": builtin_norm_1,
        "<builtin>norm_2": builtin_norm_2,
        "<builtin>norm_inf": builtin_norm_inf,
        "<builtin>dot_product": builtin_dot_product,
        "<builtin>array": builtin_array,
        "<builtin>matmul": builtin_matmul,
        "<builtin>transpose": builtin_transpose,
        "<builtin>linear_solve": builtin_linear_solve,
        "<builtin>svd": builtin_svd,
        "<builtin>print": builtin_print,
        }
