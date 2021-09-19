
import numpy as np

import pytest

from pyro.dynamic import StateSpaceSystem, linearize

from pyro.dynamic.pendulum import SinglePendulum

from pyro.dynamic.manipulator import TwoLinkManipulator

class SdofOscillator(StateSpaceSystem):
    """Single DOF mass-spring-damper system

    Defined using state-space matrices

    """

    def __init__(self, m, k, b):
        """
        m: mass
        k: spring constant
        b: damping constant
        """

        self.mass = m
        self.k = k
        self.b = b

        A = np.array([[0, 1], [-k/m, -b/m]])
        B = np.array([0, 1/m]).reshape((2, 1))

        C = np.identity(2)
        D = np.zeros((2, 1))

        super().__init__(A, B, C, D)

    def solve_analytical_free(self, x0, dx0, t):
        """
        Free vibration equations from:
        Rao (2019) Vibrations of Continuous Systems (2 ed)
        eq 2.16 - 2.21

        x0: initial position
        dx0: initial velocity
        """

        # undamped natural frequency
        wn = np.sqrt(self.k / self.mass)

        # damping ratio
        xi = self.b / 2 / self.mass / wn

        if xi < 1:
            wd = np.sqrt(1 - xi**2) * wn
            return np.exp(-xi * wn * t) *\
                 ((x0 * np.cos(wd * t)) + (((dx0 + xi*wn*x0) / wd) * np.sin(wd*t)))
        else:
            raise NotImplementedError("Only underdamped implemented")

    def solve_analytical_forced(self, x0, dx0, f0, w0, t):
        """
        Solution of *undamped* oscillator with harmonic excitation of the form

        F = f0*cos(w0*t)

        ref: Rao (2019) Vibrations of Continuous Systems (2 ed)
        eq 2.27
        """

        if self.b != 0:
            raise NotImplementedError("Only implemented for undamped system")

        wn = np.sqrt(self.k / self.mass)
        return (x0 - (f0 / (self.k - self.mass * w0**2))) * np.cos(wn*t) \
               + dx0 / wn * np.sin(wn*t) \
               + (f0 / (self.k - self.mass * w0**2)) * np.cos(w0*t)

def test_1dof_oscillator_free():
    sys = SdofOscillator(2.0, 100.0, 10.0)
    x0 = np.array([0.2, 0])

    # Simulate time solution
    sim = sys.compute_trajectory(x0)

    # Reference analytical time solution
    ref = sys.solve_analytical_free(x0[0], x0[1], sim.t)

    assert np.allclose(sim.x[:, 0], ref)
    assert np.allclose(sim.y[:, 0], ref)

def test_1dof_oscillator_forced():
    sys = SdofOscillator(m=2.0, k=100.0, b=0.0)
    x0 = np.array([0.2, 0])

    # External force
    w0 = 2
    f0 = 4.5
    def u(t):
        t = np.asarray(t)
        return (np.cos(w0*t) * f0).reshape((t.size, 1))

    # Simulate time solution
    sim = sys.compute_trajectory(x0, u=u, n=1000)

    # Reference analytical time solution
    ref = sys.solve_analytical_forced(x0[0], x0[1], f0, w0, sim.t)

    assert np.allclose(sim.x[:, 0], ref, atol=1e-5, rtol=0)
    assert np.allclose(sim.y[:, 0], ref, atol=1e-5, rtol=0)
    assert np.allclose(sim.u, u(sim.t))

def test_dimension_checks():
    # Valid dimensions for system with 5 states, 2 inputs and 4 outputs
    A = np.zeros((5, 5))
    B = np.zeros((5, 2))
    C = np.zeros((4, 5))
    D = np.zeros((4, 2))

    sys = StateSpaceSystem(A, B, C, D)
    assert sys.n == 5
    assert sys.m == 2
    assert sys.p == 4

    # Unsquare A matrix
    with pytest.raises(ValueError):
        StateSpaceSystem(A[:4, :], B, C, D)

    # B does not match number of states
    with pytest.raises(ValueError):
        StateSpaceSystem(A, B[:4, :], C, D)

    # C does not match number of states
    with pytest.raises(ValueError):
        StateSpaceSystem(A, B, C[:, :4], D)

    # mismatch number of inputs between B and D
    with pytest.raises(ValueError):
        StateSpaceSystem(A, B[:, :1], C, D)

    # mismatch number of outputs between C and D
    with pytest.raises(ValueError):
        StateSpaceSystem(A, B, C, D[:2, :])

def test_linearize_identity():
    """Linearization of linear system should be identical"""

    # ensure repeatable random matrices
    np.random.seed(0)

    A = np.random.rand(5, 5)
    B = np.random.rand(5, 2)
    C = np.random.rand(4, 5)
    D = np.random.rand(4, 2)

    sys = StateSpaceSystem(A, B, C, D)

    x0 = np.random.rand(5, 1)
    u0 = np.random.rand(2, 1)

    linearized = linearize(sys, x0, u0, 1e-3)

    assert np.allclose(sys.A, linearized.A)
    assert np.allclose(sys.B, linearized.B)
    assert np.allclose(sys.C, linearized.C)
    assert np.allclose(sys.D, linearized.D)

def test_linearize_pendulum():
    class Cartesian1Pendulum(SinglePendulum):
        """Single pendulum with cartesian coordinate ouput (y)"""
        def h(self, x, u, t):
            return np.array([np.sin(x[0]), -np.cos(x[0])]) * self.l1

    nlsys = Cartesian1Pendulum()
    nlsys.lc1 = 0.3
    nlsys.l1 = nlsys.lc1
    nlsys.m1 = 1.2
    nlsys.d1 = 0.1

    # Linearization point
    x0 = np.zeros((2, 1))
    u0 = np.zeros((1,))
    y0 = nlsys.h(x0, 0, 0).reshape((2,))

    # linearize with epsilon = 0.01 rad (~1 deg)
    linsys = linearize(nlsys, x0, u0, epsilon_x=0.01)

    # Simulate with 5 degree initial position and zero velocity
    x_init = np.array([0.087, 0])

    nlsim = nlsys.compute_trajectory(x_init, tf=10)
    linsim = linsys.compute_trajectory(x_init, tf=10)

    rtol = 0.0
    atol = x_init[0] * 0.02 # 2% error tolerance

    assert np.allclose(nlsim.t, linsim.t)
    assert np.allclose(nlsim.x, linsim.x, rtol, atol)
    assert np.allclose(nlsim.dx, linsim.dx, rtol, atol)
    assert np.allclose(nlsim.u, linsim.u)
    assert np.allclose(nlsim.y, (linsim.y + y0), rtol, atol)

if __name__ == "__main__":
    pass