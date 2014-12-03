from leap.method.im_euler import ImplicitEulerMethod
from leap.vm.implicit import GenericNumpySolver
from leap.vm.exec_numpy import NumpyInterpreter

class NewtonSolver(GenericNumpySolver):

    def run_solver(self, function, guess):
        from scipy.optimize import newton
        return newton(function, guess)

    def solve(self, expression, solve_component, context, functions, guess):
        print('Expression to solve: ' + str(expression))
        print('Solve component: ' + str(solve_component))
        print('Mapping:')
        from leap.vm.utils import get_variables
        for variable_name in get_variables(expression):
            if variable_name == solve_component.name:
                continue
            print('  ' + variable_name +  ' --> ' + str(context[variable_name]))
        return GenericNumpySolver.solve(self, expression, solve_component,
                                        context, functions, guess)

def run():
    code = ImplicitEulerMethod()('y')
    def rhs(t, y):
        return -10.0 * y
    interpreter = NumpyInterpreter(code,
                                   function_map={'<func>y': rhs},
                                   solver_map={'newton': NewtonSolver()})
    interpreter.set_up(t_start=0, dt_start=0.5, context={'y': 3.0})
    interpreter.initialize()
    for event in interpreter.run(t_end=0.5):
        pass

if __name__ == '__main__':
    run()
