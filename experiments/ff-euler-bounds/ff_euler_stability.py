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
    s_0 = np.identity(2, dtype=np.complex_)
    s_1 = STEP_RATIO * jacobian
    s_2 = np.zeros((2, 2), dtype=np.complex_)
    s_2[1, :] = [jacobian[1, 1] * jacobian[1, 0], jacobian[1, 1] ** 2.0]
    return (s_0, s_1, s_2)


def make_ff_euler_step_matrix(jacobian, dt):
    """Return the 2x2 step matrix for fastest first forward Euler."""
    s_0, s_1, s_2 = make_ff_euler_step_coefficients(jacobian)
    return s_0 + dt * s_1 + dt ** 2 * s_2


def estimate_stable_dt(jacobian, eigvals, eigvects):
    """
    Estimate a timestep that should be stable for fastest first forward Euler.

    Inputs:
    jacobian: The 2x2 Jacobian matrix of the system
    eigvals:  The eigenvalues of the Jacobian
    eigvects: The eigenvectors the Jacobian

    Returns:
    If a stable timestep is calculated, then the timestep is returned.
    Otherwise, None is returned.
    """

    s_0, s_1, s_2 = make_ff_euler_step_coefficients(jacobian)

    dt = np.inf

    for eigval in eigvals:
        lambda_re = np.real(2 * eigval)
        lambda_norm = np.abs(2 * eigval)

        jacobian_eigvect_cond = la.cond(eigvects, p=2)
        perturbation_norm = la.norm(s_2, ord=2)

        def delta(h):
            return jacobian_eigvect_cond * h * perturbation_norm

        def upper_bound(h):
            return -(lambda_re + delta(h)) / ((lambda_norm + delta(h)) ** 2) - h

        try:
            dt = np.min([dt, opt.newton(upper_bound, 0.1)])
        except:
            return None

    return dt


def is_stable(jacobian, dt):
    system = np.array([1.0, 1.0])
    system /= la.norm(system)
    step_matrix = make_ff_euler_step_matrix(jacobian, dt)
    for i in range(0, 100):
        system = step_matrix.dot(system)
    return la.norm(system) < 1.0


MAX_STEP_SIZE = 5.0


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


def computational_stable_dt(jacobian):
    """Estimate the stable timestep by determining the largest timestep such
    that spectral radius of the step matrix is under 1."""
    def radius(h):
        step_matrix = make_ff_euler_step_matrix(jacobian, h)
        return la.norm(la.eigvals(step_matrix), ord=np.inf) - 1.0
    try:
        return opt.bisect(radius, 1.0e-16, MAX_STEP_SIZE)
    except:
        return None


def make_test_matrices():
    LAMBDA_SPACING = 10
    THETA_SPACING = 20
    OFFSET_SPACING = 10
    lambda_2 = -1.0
    for lambda_1 in np.linspace(-1, -0.1, LAMBDA_SPACING):
        for alpha in np.linspace(0, np.pi, THETA_SPACING):
            for beta in np.linspace(np.pi / OFFSET_SPACING, np.pi, OFFSET_SPACING):
                eigvects = np.array([[np.cos(alpha), np.cos(alpha + beta)],
                                     [np.sin(alpha), np.sin(alpha + beta)]])
                eigvals_diag = np.array([[lambda_1, 0.0], [0.0, lambda_2]])
                try:
                    result = eigvects.dot(eigvals_diag.dot(la.inv(eigvects)))
                    yield (result, [lambda_1, lambda_2], eigvects)
                except:
                    continue


def collect_data():
    num_total_tests = 0
    num_skipped_tests = 0
    num_estimate_failures = 0
    timestep_relative_errors = []
    spectral_radiuses_at_estimates = []

    for jacobian, eigvals, eigvects in make_test_matrices():
        num_total_tests += 1
        dt_experimental = experimental_stable_dt(jacobian)
        dt_computational = computational_stable_dt(jacobian)
        if not dt_computational and not dt_experimental:
            num_skipped_tests += 1
            continue
        if not dt_computational:
            dt_true = dt_experimental
        elif not dt_experimental:
            dt_true = dt_computational
        else:
            dt_true = np.mean([dt_experimental, dt_computational])
        dt_estimate = estimate_stable_dt(jacobian, eigvals, eigvects)
        if not dt_estimate:
            num_estimate_failures += 1
            continue
        step_matrix = make_ff_euler_step_matrix(jacobian, dt_estimate)
        radius = la.norm(la.eigvals(step_matrix), ord=np.inf)
        if radius > 1.0:
            assert np.isclose(radius - 1.0, 0, atol=1.0e-3)
        relative_error = (dt_estimate - dt_true) / dt_true
        if not np.isfinite(relative_error) or not \
                np.isfinite(radius) or dt_estimate > dt_true:
            num_skipped_tests += 1
            continue
        timestep_relative_errors.append(relative_error)
        spectral_radiuses_at_estimates.append(radius)

    print('Number of total tests: ' + str(num_total_tests))
    print('Number of tests skipped due to computational failure: '
          + str(num_skipped_tests))
    print('Number of tests skipped due to inability to estimate timestep: '
          + str(num_estimate_failures))
    print('Mean relative error in estimate: '
          + str(np.mean(timestep_relative_errors)))
    print('Mean spectral radius at estimate: ' +
          str(np.mean(spectral_radiuses_at_estimates)))


if __name__ == '__main__':
    collect_data()
