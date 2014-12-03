from leap.method.im_euler import ImplicitEulerMethod
from leap.vm.implicit import GenericNumpySolver
from leap.vm.exec_numpy import NumpyInterpreter

class NewtonSolver(GenericNumpySolver):

    def __init__(self):
        self.printed = False

    def run_solver(self, function, guess):
        from scipy.optimize import newton
        return newton(function, guess)

    def solve(self, expression, solve_component, context, functions, guess):
        if not self.printed:
            print('Expression to solve: ' + str(expression))
            print('Solve component: ' + str(solve_component))
            self.printed = True
        return GenericNumpySolver.solve(self, expression, solve_component,
                                        context, functions, guess)

def run():
    code = ImplicitEulerMethod()('y')
    def rhs(t, y):
        return -10.0 * y
    interpreter = NumpyInterpreter(code, function_map={'<func>y': rhs},
                                   solver_map={'newton': NewtonSolver()})
    interpreter.set_up(0, 0.1, context={'y': 1.0})
    interpreter.initialize()
    for event in interpreter.run(1.0):
        pass

if __name__ == '__main__':
    run()
