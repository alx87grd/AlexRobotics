"""
Run all examples

Does not check for correct results, only that there are no exceptions.

Running this:

* Using pytest: `pytest -ra examples/` (replace examples with path to the
  examples folder where this script lives). Pytest will produce a report
  of failed/succeeded tests

* As a script: `python ./examples/test_all_examples.py`. Will stop at the
  exception.

TODO: Add actual tests that check for correct results

"""

from pathlib import Path

from importlib.util import spec_from_file_location, module_from_spec

import sys

import inspect

from matplotlib import pyplot as plt

_all_examples = [
    "./simple_pendulum/custom_simple_pendulum.py",
    "./simple_pendulum/simple_pendulum_with_computed_torque.py",
    "./simple_pendulum/simple_pendulum_with_open_loop_controller.py",
    "./simple_pendulum/simple_pendulum_with_pid.py",
    "./simple_pendulum/simple_pendulum_with_rrt.py",
    "./simple_pendulum/simple_pendulum_with_sliding_mode_controller.py",
    "./simple_pendulum/simple_pendulum_with_trajectory_following_computed_torque.py",
    "./simple_pendulum/simple_pendulum_with_trajectory_following_sliding_mode_controller.py",
    "./simple_pendulum/simple_pendulum_with_valueiteration.py"
]

this_script_dir = Path(__file__).parent
this_module = sys.modules[__name__]
#print(_all_examples)

def import_from_file(modulename, filepath):
    """import a file as a module and return the module object

    Everything will be executed, except if it's conditional to
     __name__ == "__main__"

    """

    spec = spec_from_file_location(modulename, filepath)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod

def gettestfunc(modname, fullpath):
    """Create a function that imports a file and runs the main function

    """

    def run_example_main():
        ex_module = import_from_file(modname, fullpath)

        # Call function `main()` if it exists
        if hasattr(ex_module, "main"):
            ex_main_fun = getattr(ex_module, "main")
            ex_main_fun()

        # Close all figs to reclaim memory
        plt.close('all')

    return run_example_main

_all_test_funs = []

for example_file in _all_examples:
    relpath = Path(example_file)
    fullpath = this_script_dir.joinpath(relpath)
    modname = relpath.stem # file name without extension

    # Define a new function with a name starting with test_ so it is ran
    # by pytest.
    setattr(this_module, "test_" + modname,
        gettestfunc(modname, fullpath))

def main():
    for fun in _all_test_funs:
        fun()

if __name__ == "__main__":
    main()