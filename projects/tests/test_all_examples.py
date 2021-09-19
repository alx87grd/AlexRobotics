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
    "./bicycle/bicycle.py",
    "./bicycle/bicycle_exploration_with_rrt.py", 
    "./bicycle/bicycle_parallel_parking_with_rrt.py",
    "./cartpole/cartpole_natural_behavior.py",
    "./cartpole/cartpole_with_computed_torque.py",
    "./cartpole/cartpole_with_rrt_and_computed_torque.py",
    "./cartpole/underactuated_cartpole_swingup.py",
    "./cartpole/underactuated_cartpole_with_partialfeedbacklinearization.py",
    "./cartpole/underactuated_cartpole_with_rrt.py",
    "./double_pendulum/double_pendulum.py",
    "./double_pendulum/double_pendulum_with_computed_torque.py",
    "./double_pendulum/double_pendulum_with_rrt.py",
    "./double_pendulum/double_pendulum_with_rrt_and_computed_torque.py",
    "./double_pendulum/double_pendulum_with_sliding_mode.py",
    "./double_pendulum/double_pendulum_with_trajectory_following_computed_torque.py",
    "./double_pendulum/double_pendulum_with_trajectory_following_open_loop_controller.py",
    "./double_pendulum/double_pendulum_with_trajectory_following_sliding_mode_controller.py",
    "./holonomic_mobile_robot/holonomic_mobile_robot_exploration_with_obstacles_with_rrt.py",
    "./holonomic_mobile_robot/holonomic_mobile_robot_exploration_with_rrt.py",
    "./holonomic_mobile_robot/holonomic_mobile_robot_with_valueiteration.py",
    "./integrators/double_integrator.py",
    "./integrators/double_integrator_optimal_controller.py",
    "./integrators/integrators_with_closed_loops.py",
    "./integrators/simple_integrator.py",
    "./integrators/triple_integrator.py",
    "./robot_arms/fivelinkrobot_kinematic_controller.py",
    "./robot_arms/threelinkrobot_computed_torque_controller.py",
    "./robot_arms/threelinkrobot_kinematic_controller.py",
    "./robot_arms/twolinkrobot_computed_torque_controller.py",
    "./robot_arms/twolinkrobot_effector_impedance_controller.py",
    "./robot_arms/twolinkrobot_joint_impedance_controller.py",
    "./robot_arms/twolinkrobot_kinematic_controller.py",
    "./robot_arms/twolinkrobot_kinematic_vs_dynamic_openloop.py",
    "./robot_arms/twolinkrobot_quasi_static_controllers.py",
    "./simple_pendulum/custom_simple_pendulum.py",
    "./simple_pendulum/simple_pendulum_with_computed_torque.py",
    "./simple_pendulum/simple_pendulum_with_open_loop_controller.py",
    "./simple_pendulum/simple_pendulum_with_pid.py",
    "./simple_pendulum/simple_pendulum_with_rrt.py",
    "./simple_pendulum/simple_pendulum_with_sliding_mode_controller.py",
    "./simple_pendulum/simple_pendulum_with_trajectory_following_computed_torque.py",
    "./simple_pendulum/simple_pendulum_with_trajectory_following_sliding_mode_controller.py",
    "./simple_pendulum/simple_pendulum_with_valueiteration.py",
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