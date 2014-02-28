from __future__ import division
from pytools import Record
import numpy
import scipy.linalg as la
from pymbolic import expand, differentiate
from pymbolic.mapper.constant_folder import \
        CommutativeConstantFoldingMapper

def fold_constants(expr):
    return CommutativeConstantFoldingMapper()(expr)




# {{{ utilities ---------------------------------------------------------------
class FactoryWithParameters(Record):
    __slots__ = []

    def get_parameter_dict(self):
        result = {}
        for f in self.__class__.fields:
            try:
                result[intern(f)] = getattr(self, f)
            except AttributeError:
                pass
        return result

class StabilityTester(object):
    def __init__(self, method_fac, matrix_fac):
        self.method_fac = method_fac
        self.matrix_fac = matrix_fac

    @property
    def matrix(self):
        try:
            return self.matrix_cache
        except AttributeError:
            self.matrix_cache = self.matrix_fac()
            return self.matrix_cache

    def get_parameter_dict(self):
        result = {}
        result.update(self.method_fac.get_parameter_dict())
        result.update(self.matrix_fac.get_parameter_dict())
        return result

    def refine(self, stable, unstable):
        assert self.is_stable(stable)
        assert not self.is_stable(unstable)
        while abs(stable-unstable) > self.prec:
            mid = (stable+unstable)/2
            if self.is_stable(mid):
                stable = mid
            else:
                unstable = mid
        else:
            return stable

    def find_stable_dt(self):
        dt = 0.1

        if self.is_stable(dt):
            dt *= 2
            while self.is_stable(dt):
                dt *= 2

                if dt > 2**8:
                    return dt
            return self.refine(dt/2, dt)
        else:
            dt /= 2
            while not self.is_stable(dt):
                dt /= 2

                if dt < self.prec:
                    return dt
            return self.refine(dt, dt*2)

    def __call__(self):
        return { "dt": self.find_stable_dt() }




class IterativeStabilityTester(StabilityTester):
    def __init__(self, method_fac, matrix_fac, stable_steps):
        StabilityTester.__init__(self, method_fac, matrix_fac)
        self.stable_steps = stable_steps

    def get_parameter_dict(self):
        result = StabilityTester.get_parameter_dict(self)
        result["stable_steps"] = self.stable_steps
        return result

# }}}

# {{{ matrices ----------------------------------------------------------------
class MatrixFactory(FactoryWithParameters):
    __slots__ = ["ratio", "angle", "offset"]

    def get_parameter_dict(self):
        res = FactoryWithParameters.get_parameter_dict(self)
        res["mat_type"] = type(self).__name__
        return res

    def get_eigvec_mat(self):
        from math import cos, sin
        return numpy.array([
            [cos(self.angle), cos(self.angle+self.offset)],
            [sin(self.angle), sin(self.angle+self.offset)],
            ])

    def __call__(self):
        mat = numpy.diag([-1, -1*self.ratio])
        evmat = self.get_eigvec_mat()
        return numpy.dot(la.solve(evmat, mat), evmat)




class DecayMatrixFactory(MatrixFactory):
    __slots__ = []

    def __call__(self):
        vec = numpy.array([-1, -1*self.ratio])
        mat = numpy.diag(vec)
        evmat = self.get_eigvec_mat()
        return numpy.dot(la.solve(evmat, mat), evmat)

class DecayOscillationMatrixFactory(MatrixFactory):
    __slots__ = []

    def __call__(self):
        vec = numpy.array([-1, 1j*self.ratio])
        mat = numpy.diag(vec)
        evmat = self.get_eigvec_mat()
        return numpy.dot(la.solve(evmat, mat), evmat)

class OscillationDecayMatrixFactory(MatrixFactory):
    __slots__ = []

    def __call__(self):
        vec = numpy.array([1j, -1*self.ratio])
        mat = numpy.diag(vec)
        evmat = self.get_eigvec_mat()
        return numpy.dot(la.solve(evmat, mat), evmat)

class OscillationMatrixFactory(MatrixFactory):
    __slots__ = []

    def __call__(self):
        vec = numpy.array([1j, 1j*self.ratio])
        mat = numpy.diag(vec)
        evmat = self.get_eigvec_mat()
        from hedge.tools.linalg import leftsolve
        return numpy.dot(evmat, leftsolve(evmat, mat))



def generate_matrix_factories():
    from math import pi

    angle_steps = 20
    offset_steps = 20
    for angle in numpy.linspace(0, pi, angle_steps, endpoint=False):
        for offset in numpy.linspace(
                pi/offset_steps, 
                pi, offset_steps, endpoint=False):
            for ratio in numpy.linspace(0.1, 1, 10):
                yield DecayMatrixFactory(ratio=ratio, angle=angle, offset=offset)
                yield OscillationMatrixFactory(ratio=ratio, angle=angle, offset=offset)
                yield OscillationDecayMatrixFactory(ratio=ratio, angle=angle, offset=offset)
                yield DecayOscillationMatrixFactory(ratio=ratio, angle=angle, offset=offset)





def generate_matrix_factories_hires():
    from math import pi

    offset_steps = 100
    for angle in [0, 0.05*pi, 0.1*pi]:
        for offset in numpy.linspace(
                pi/offset_steps, 
                pi, offset_steps, endpoint=False):
            for ratio in numpy.linspace(0.1, 1, 100):
                yield DecayMatrixFactory(ratio=ratio, angle=angle, offset=offset)
                yield OscillationMatrixFactory(ratio=ratio, angle=angle, offset=offset)
                yield OscillationDecayMatrixFactory(ratio=ratio, angle=angle, offset=offset)
                yield DecayOscillationMatrixFactory(ratio=ratio, angle=angle, offset=offset)

# }}}

# {{{ MRAB --------------------------------------------------------------------

def make_method_matrix(stepper, rhss, f_size, s_size):
    from pymbolic import var

    from pytools.obj_array import make_obj_array
    f = make_obj_array([
        var("f%d" % i) for i in range(f_size)])
    s = make_obj_array([
        var("s%d" % i) for i in range(s_size)])

    from hedge.timestep.multirate_ab.methods import (
            HIST_F2F, HIST_S2F, HIST_F2S, HIST_S2S,
            HIST_NAMES)

    hist_sizes = {
                HIST_F2F: f_size,
                HIST_S2F: f_size,
                HIST_S2S: s_size,
                HIST_F2S: s_size
                }

    orig_histories = {}
    for hn in HIST_NAMES:
        my_size = hist_sizes[hn]
        my_length = stepper.orders[hn]
        hist_name_str = hn.__name__[5:].lower()
        hist = [
                make_obj_array([
                    var("h_%s_%d_%d" % (hist_name_str, age, i) )
                    for i in range(s_size)])
                for age in range(my_length)]

        stepper.histories[hn] = hist
        orig_histories[hn] = hist[:]

    stepper.startup_stepper = None
    del stepper.startup_history

    f_post_step, s_post_step = stepper([f, s], 0, rhss)

    def matrix_from_expressions(row_exprs, column_exprs):
        row_exprs = [fold_constants(expand(expr))
                for expr in row_exprs]

        result = numpy.zeros((len(row_exprs), len(column_exprs)),
                dtype=object)

        for i, row_expr in enumerate(row_exprs):
            for j, col_expr in enumerate(column_exprs):
                result[i,j] = differentiate(row_expr, col_expr)

        return result

    from pytools import flatten
    row_exprs = list(flatten([
        f_post_step, 
        s_post_step,
        ] + list(flatten([stepper.histories[hn] for hn in HIST_NAMES]))
        ))

    column_exprs = list(flatten([
        f, 
        s
        ] + list(flatten([orig_histories[hn] for hn in HIST_NAMES]))
        ))

    return matrix_from_expressions(row_exprs, column_exprs), \
            column_exprs





class MethodFactory(FactoryWithParameters):
    __slots__ = ["method", "substep_count", "meth_order"]

    def __call__(self, dt):
        from hedge.timestep.multirate_ab import \
                TwoRateAdamsBashforthTimeStepper
        return TwoRateAdamsBashforthTimeStepper(
                method=self.method,
                large_dt=dt,
                substep_count=self.substep_count,
                order=self.meth_order)

def generate_method_factories():
    from hedge.timestep.multirate_ab.methods import methods

    if True:
        for method in methods.keys():
            for order in [3]:
                for substep_count in [2, 3, 4]:
                    yield MethodFactory(method=method, meth_order=order, 
                            substep_count=substep_count)
    else:
        for method in ["Fqsr"]:
            for order in [3]:
                for substep_count in [2, 3]:
                    yield MethodFactory(method=method, meth_order=order, 
                            substep_count=substep_count)




def generate_method_factories_hires():
    for method in ["Fq", "Ssf", "Sr"]:
        for order in [3]:
            for substep_count in [2, 5, 10]:
                yield MethodFactory(method=method, meth_order=order, 
                        substep_count=substep_count)






class IterativeMRABJob(IterativeStabilityTester):
    prec = 1e-4

    def is_stable(self, dt):
        stepper = self.method_fac(dt)
        mat = self.matrix

        y = numpy.array([1,1], dtype=numpy.float64)
        y /= la.norm(y)

        def f2f_rhs(t, yf, ys): return mat[0,0] * yf()
        def s2f_rhs(t, yf, ys): return mat[0,1] * ys()
        def f2s_rhs(t, yf, ys): return mat[1,0] * yf()
        def s2s_rhs(t, yf, ys): return mat[1,1] * ys()

        for i in range(self.stable_steps):
            y = stepper(y, i*dt, 
                    (f2f_rhs, s2f_rhs, f2s_rhs, s2s_rhs))
            if la.norm(y) > 10:
                return False

        return True




class DirectMRABJob(StabilityTester):
    prec = 1e-8

    @property
    def method_matrix_func(self):
        try:
            return self.method_matrix_func_cache
        except AttributeError:
            from pymbolic import var
            stepper = self.method_fac(var("dt"))

            mat = self.matrix

            def f2f_rhs(t, yf, ys): 
                return fold_constants(expand(mat[0,0] * yf()))
            def s2f_rhs(t, yf, ys): 
                return fold_constants(expand(mat[0,1] * ys()))
            def f2s_rhs(t, yf, ys): 
                return fold_constants(expand(mat[1,0] * yf()))
            def s2s_rhs(t, yf, ys): 
                return fold_constants(expand(mat[1,1] * ys()))

            method_matrix, _ = make_method_matrix(stepper, 
                    rhss=(f2f_rhs, s2f_rhs, f2s_rhs, s2s_rhs),
                    f_size=1, s_size=1)

            from pymbolic import compile
            self.method_matrix_func_cache = compile(method_matrix)
            return self.method_matrix_func_cache 

    def is_stable(self, dt):
        eigvals = la.eigvals(self.method_matrix_func(dt))
        max_eigval = numpy.max(numpy.abs(eigvals))
        return max_eigval <= 1




def generate_mrab_jobs():
    for method_fac in generate_method_factories():
        for matrix_fac in generate_matrix_factories():
            yield DirectMRABJob(method_fac, matrix_fac)




def generate_mrab_jobs_hires():
    for method_fac in generate_method_factories_hires():
        for matrix_fac in generate_matrix_factories_hires():
            yield DirectMRABJob(method_fac, matrix_fac)




def generate_mrab_jobs_step_verify():
    from math import pi

    def my_generate_matrix_factories():

        offset_steps = 20
        for angle in [0.05*pi]:
            for offset in numpy.linspace(
                    pi/offset_steps, 
                    pi, offset_steps, endpoint=False):
                for ratio in numpy.linspace(0.1, 1, 10):
                    yield DecayMatrixFactory(ratio=ratio, angle=angle, offset=offset)
                    yield OscillationMatrixFactory(ratio=ratio, angle=angle, offset=offset)
                    yield OscillationDecayMatrixFactory(ratio=ratio, angle=angle, offset=offset)
                    yield DecayOscillationMatrixFactory(ratio=ratio, angle=angle, offset=offset)

    for method_fac in list(generate_method_factories())[:1]:
        for matrix_fac in my_generate_matrix_factories():
            for stable_steps in [40, 80, 120]:
                yield DirectMRABJob(method_fac, matrix_fac, stable_steps)

# }}}




# {{{ single-rate reference ---------------------------------------------------
class SRABMethodFactory(FactoryWithParameters):
    __slots__ = ["method", "substep_count", "meth_order"]

    def __call__(self):
        from hedge.timestep.ab import AdamsBashforthTimeStepper
        return AdamsBashforthTimeStepper(order=self.meth_order,
                dtype=numpy.complex128)




class SRABJob(IterativeStabilityTester):
    prec = 1e-4

    def is_stable(self, dt):
        stepper = self.method_fac()

        y = numpy.array([1,1], dtype=numpy.complex128)
        y /= la.norm(y)

        def rhs(t, y):
            return numpy.dot(self.matrix, y)

        for i in range(self.stable_steps):
            y = stepper(y, i*dt, dt, rhs)
            if la.norm(y) > 10:
                return False

        return True




def generate_srab_jobs():
    for method_fac in [SRABMethodFactory(method="SRAB", substep_count=1, meth_order=3)]:
        for matrix_fac in generate_matrix_factories():
            yield SRABJob(method_fac, matrix_fac, 120)

# }}}




def test():
    from hedge.timestep.multirate_ab import \
            TwoRateAdamsBashforthTimeStepper
    from pymbolic import var
    stepper = TwoRateAdamsBashforthTimeStepper(
            method="Fqsr",
            large_dt=var("dt"),
            substep_count=2,
            order=1)

    mat = numpy.random.randn(2,2)

    def f2f_rhs(t, yf, ys): 
        return fold_constants(expand(mat[0,0] * yf()))
    def s2f_rhs(t, yf, ys): 
        return fold_constants(expand(mat[0,1] * ys()))
    def f2s_rhs(t, yf, ys): 
        return fold_constants(expand(mat[1,0] * yf()))
    def s2s_rhs(t, yf, ys): 
        return fold_constants(expand(mat[1,1] * ys()))

    z, vars = make_method_matrix(stepper, 
            rhss=(f2f_rhs, s2f_rhs, f2s_rhs, s2s_rhs),
            f_size=1, s_size=1)

    from pymbolic import compile
    num_mat_func = compile(z)
    num_mat = num_mat_func(0.1)

    if False:
        from pymbolic import substitute
        num_mat_2 = numpy.array(
                fold_constants(substitute(z, dt=0.1)),
                dtype=numpy.complex128)

        print la.norm(num_mat-num_mat_2)

    if True:
        for row, var in zip(num_mat, vars):
            print "".join("*" if entry else "." for entry in row), var





if __name__ == "__main__":
    #test()
    run()

# vim: foldmethod=marker
