
import numpy as np

from scipy import integrate, signal

import pytest

from pyro.control.linear import PIDController, ProportionalController

from pyro.dynamic.system import ContinuousDynamicSystem

from pyro.dynamic.manipulator import TwoLinkManipulator

class FirstOrder(ContinuousDynamicSystem):
    def __init__(self, tau):
        super().__init__(1, 1, 1)
        self.tau = tau

    def f(self, x, u, t):
        return (u - x) / self.tau

def step(a, delay=0):
    def u(t):
        if t < delay:
            return 0
        else:
            return a
    return u

def filtered_deriv(y, x, tau=0):
    """Numerical derivative with optional lowpass filter.

    tau is the filter time constant expressed in same units as x (eg seconds if x is
    time).
    """
    dy = np.empty(y.shape)
    dy[0] = (y[1] - y[0]) / (x[1] - x[0])
    for i in range(1, (dy.shape[0] - 1)):
        dy[i] = (y[i+1] - y[i-1]) / (x[i+1] - x[i-1])
    dy[-1] = (y[-1] - y[-2]) / (x[-1] - x[-2])

    # No filter
    if tau <= 0:
        return dy

    # sample freq
    fs = x.shape[0] / (x[-1] - x[0])
    nyqfreq = fs / 2
    w0 = 1 / (tau*2*np.pi*nyqfreq)

    lowpass = signal.iirfilter(1, w0, btype='lowpass', analog=False)

    # Create initial conditions so y=0 at t=0
    zi = signal.lfiltic(*lowpass, y=[0], x=dy[:1])

    filtered, _ = signal.lfilter(*lowpass, dy, zi=-zi)
    return filtered

def test_sdof_prop():
    """Test single DOF system with proportional control"""
    tau_p = 2
    tf = 1.5
    kp = 20

    sys = FirstOrder(tau_p)
    ctl = ProportionalController(20)
    clsys = ctl + sys

    sim = clsys.compute_trajectory(x0=[0], r=step(1), tf=tf, n=100)

    # analytic solution
    kcl = kp / (kp + 1) # steady state asymptote value
    tau_cl = tau_p / (kp + 1) # Closed loop time constant
    x_ref = kcl * (1-np.exp(-sim.t/tau_cl))

    assert np.allclose(sim.x[:, 0], x_ref)

def test_sdof_pid():
    """Check PID controller outputs"""
    tf = 2 # simulation time
    npts = 500 # simulation samples
    tau = 1E-2 # time constant of derivative filter

    # Create system/controller and run simulation
    sys = FirstOrder(1)
    ctl = PIDController([[3]], [[2]], [[0.6]], dv_tau=tau)
    clsys = ctl + sys
    sim = clsys.compute_trajectory(0, r=step(1), tf=tf, n=npts)

    # error
    e = (sim.r - sim.y)[:, 0]

    # integral of error
    e_int = integrate.cumtrapz(e, sim.t, initial=0)

    # filtered derivative of error
    e_fd = filtered_deriv(e, sim.t, tau)

    # Check controller output
    ctl_ref = (e * ctl.KP + e_int * ctl.KI + e_fd * ctl.KD).reshape((sim.n,))

    assert np.allclose(sim.u[:, 0], ctl_ref, atol=0, rtol=1E-2)

def test_nd_pid():
    tf = 2
    npts = 500
    tau = 1E-2

    class DummyManipulator(TwoLinkManipulator):
        """Manipulator with null response to inputs"""
        def __init__(self):
            super().__init__()
            self.p = 2

        def B(self, q):
            return np.zeros(self.dof)

        def h(self, x, u, t):
            """Read joint positions"""
            return x[:2]

    sys = DummyManipulator()
    ctl = PIDController(
        KP = np.ones((2,2)),
        KI = np.ones((2,2)),
        KD = np.ones((2,2)),
        dv_tau=tau
    )
    clsys = ctl + sys

    x0 = [np.pi-0.5, np.pi/2, 0, 0]
    sim = clsys.compute_trajectory(x0=x0, tf=tf, n=npts)

    errors = sim.r - sim.y

    e_int = integrate.cumtrapz(errors, sim.t, axis=0, initial=0)

    e_der = np.empty(errors.shape)
    for j in range(errors.shape[1]):
        e_der[:, j] = filtered_deriv(errors[:, j], sim.t, tau=tau)

    ctl_ref = errors.dot(ctl.KP.T) + e_int.dot(ctl.KI.T) + e_der.dot(ctl.KD.T)

    assert np.allclose(sim.u, ctl_ref, atol=0.05)

def test_check_and_normalize():
    # Check normalization of scalars to 2D array
    ctl = PIDController(1, 1, 1)
    for arr in [ctl.KP, ctl.KI, ctl.KD]:
        assert arr.ndim == 2
        assert arr.shape == (1, 1)
        assert arr[0, 0] == 1

    # Check normalization of 1D array to 2D array
    vals = [1, 2, 3]
    ctl = PIDController(vals, vals, vals)
    for arr in [ctl.KP, ctl.KI, ctl.KD]:
        assert arr.ndim == 2
        assert arr.shape == (1, 3)
        assert np.all(vals == arr)

    # Check that 2D arrays should be unchanged
    vals = np.ones((3, 4)) * 2.3
    ctl = PIDController(vals, vals, vals)
    for arr in [ctl.KP, ctl.KI, ctl.KD]:
        assert arr.ndim == 2
        assert arr.shape == vals.shape
        assert np.all(vals == arr)

    # Check default zero values for KI and KD
    for testval in [1, [1, 2, 3], np.ones((8, 10))]:
        ctl = PIDController(testval)
        for arr in [ctl.KI, ctl.KD]:
            assert arr.shape == ctl.KP.shape
            assert np.all(arr == np.zeros(ctl.KP.shape))

    with pytest.raises(ValueError):
        ctl = PIDController(1, KI=np.ones(2))
    with pytest.raises(ValueError):
        ctl = PIDController(1, KD=np.ones(2))
    with pytest.raises(ValueError):
        ctl = PIDController(np.ones((2, 3)), KI=np.ones((3, 2)))
    with pytest.raises(ValueError):
        ctl = PIDController(np.ones((2, 3)), KD=np.ones((3, 2)))

if __name__ == "__main__":
    test_check_and_normalize()