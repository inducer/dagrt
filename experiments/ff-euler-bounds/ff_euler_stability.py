"""Test the accuracy of stable timestep estimates for 2x2 fastest first Euler
methods."""

from __future__ import division

import numpy as np
import numpy.linalg as la
import scipy.optimize as opt
from pytools import memoize


STEP_RATIO = 2


@memoize(key=lambda jacobian: tuple(jacobian.flatten()))
def make_ff_euler_step_coefficients(jacobian):
    """
    Return a triplet of matrices (s_0, s_1, s_2) such that the polynomial

      s_0 + dt * s_1 + dt**2 * s_2

    represents the step matrix for fastest-first Euler associated with dt.
    """
    assert jacobian.shape == (2, 2)
    s_0 = np.identity(2, dtype=np.complex128)
    s_1 = STEP_RATIO * jacobian
    s_2 = np.zeros((2, 2), dtype=np.complex128)
    s_2[1, :] = [jacobian[1, 1] * jacobian[1, 0], jacobian[1, 1] ** 2.0]
    return (s_0, s_1, s_2)


def make_ff_euler_step_matrix(jacobian, dt):
    """Return the 2x2 step matrix for fastest first forward Euler."""
    s_0, s_1, s_2 = make_ff_euler_step_coefficients(jacobian)
    return s_0 + dt * s_1 + dt ** 2 * s_2


def unperturbed_dt(eigvals):
    """Return the maximum stable timestep for unperturbed Euler."""
    eigvals = STEP_RATIO * np.complex128(eigvals)
    hs = -2 * eigvals.real / np.abs(eigvals) ** 2
    
    return hs.min()


def bauer_fike_dt(jacobian, eigvals, eigvects):
    """
    Estimate a timestep that should be stable for fastest first forward Euler.

    Inputs:
    jacobian: The 2x2 Jacobian matrix of the system
    eigvals:  The eigenvalues of the Jacobian
    eigvects: The eigenvectors the Jacobian

    Returns:
    The stable timestep value
    """

    s_0, s_1, s_2 = make_ff_euler_step_coefficients(jacobian)

    d = la.cond(eigvects, p=2) * la.norm(s_2, ord=2)

    dt = np.inf

    eigvals = np.complex128(eigvals)

    for eigval in STEP_RATIO * eigvals:
        l = np.abs(eigval)
        r_l = eigval.real

        h = None

        roots = np.roots([d ** 2, 2 * l * d, l ** 2 - 2 * d, 2 * r_l])
        real_roots = []
        for root in roots:
            real_if_close = np.real_if_close(root)
            if real_if_close.imag == 0:
                if real_if_close.real > 0:
                    assert h is None
                    h = real_if_close.real
        dt = np.min([h, dt])

    norm = la.norm(la.eigvals(make_ff_euler_step_matrix(jacobian, dt)),
                   ord=np.inf)
    assert norm < 1
    return dt


def is_stable(jacobian, dt):
    system = np.array([1.0, 1.0]) / np.sqrt(2)
    step_matrix = make_ff_euler_step_matrix(jacobian, dt)
    for i in range(0, 100):
        system = step_matrix.dot(system)
    return la.norm(system) <= 10.0


def computational_stable_dt(jacobian):
    """Estimate the stable timestep by determining the largest timestep such
    that spectral radius of the step matrix is under 1."""
    def radius(h):
        step_matrix = make_ff_euler_step_matrix(jacobian, h)
        return la.norm(la.eigvals(step_matrix), ord=np.inf) - 1.0
    try:
        o = opt.bisect(radius, 0.05, 1.0)
        return o
    except:
        return None


MAX_STEP_SIZE = 10.0


def experimental_stable_dt(jacobian):
    """Experimentally determine the largest timestep that should be stable for
    fastest first forward Euler."""

    prec = 1.0e-8

    dt = 0.0
    right = MAX_STEP_SIZE
    left = 0.0
    while True:
        dt = left + (right - left) / 2
        if dt < prec:
            return None
        elif dt > right - prec:
            return dt
        if is_stable(jacobian, dt):
            left = dt
            dt_prime = dt + (right - left) / 2
            if not is_stable(jacobian, dt_prime):
                if abs(dt - dt_prime) < prec:
                    return dt
                right = dt_prime
            else:
                left = dt_prime
        else:
            right = dt


def make_test_matrix(theta, beta, n, lambda_=-1.0):
    eigvects = np.array([[np.cos(theta), np.cos(theta + beta)],
                         [np.sin(theta), np.sin(theta + beta)]])
    eigvals_diag = np.array([[lambda_, 0.], [0., n * lambda_]])
    result = eigvects.dot(eigvals_diag.dot(la.inv(eigvects)))
    return (result, [lambda_, n * lambda_], eigvects)


_betas = np.pi * np.array([1 / 8, 1 / 4, 3 / 8, 1 / 2])
_ns = [2, 3, 4]

def make_test_matrices():
    thetas = np.linspace(0, np.pi, 100)
    for theta in thetas:
        for beta in _betas:
            for n in _ns:
                matrix, eigvals, eigvects = make_test_matrix(theta, beta, n)
                assert la.det(matrix) != 0
                yield (theta, beta, n, matrix, eigvals, eigvects)


def collect_data():
    from pytools import Table
    stability_data = {}
    perturbation_data = {}
    for theta, beta, n, matrix, eigvals, eigvects in make_test_matrices():
        estimated_dt = bauer_fike_dt(matrix, eigvals, eigvects)
        stable_dt = computational_stable_dt(matrix)
        assert estimated_dt < stable_dt
        unperturbed_dt_ = unperturbed_dt(eigvals)
        if (n, beta) not in stability_data:
            stability_data[n, beta] = []
            perturbation_data[n, beta] = []
        stability_data[n, beta].append(estimated_dt / stable_dt)
        perturbation_data[n, beta].append(estimated_dt / unperturbed_dt_)

    from scipy.stats import gmean
    stability_table = Table()
    perturbation_table = Table()
    for n in _ns:
        stability_row = [n]
        perturbation_row = [n]
        for beta in _betas:
            min_stab = np.min(stability_data[n, beta])
            stability_row.append("{:0.3f}".format(min_stab))
            min_perturb = np.min(perturbation_data[n, beta])
            perturbation_row.append("{:0.3f}".format(min_perturb))
        stability_table.add_row(stability_row)
        perturbation_table.add_row(perturbation_row)
    with open("h-to-computed.tex", "w") as f:
        f.write(stability_table.latex())
    with open("h-to-unperturbed.tex", "w") as f:
        f.write(perturbation_table.latex())

if __name__ == '__main__':
    collect_data()
        
