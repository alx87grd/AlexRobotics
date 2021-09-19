
from collections import namedtuple

import numpy as np

import pytest

from pyro.dynamic import integrator


SystemTestCase = namedtuple('SystemTestCase', ['sut', 'x0', 'get_ref_sol'])
SysSolution = namedtuple('SysSolution', ['t', 'x', 'y'])

# Fixtures

@pytest.fixture
def double_integ():
    """Generate a DoubleIntegrator CDS and accompanying solution
    computed from analytical solution.
    """

    # System under test
    sut = integrator.DoubleIntegrator()
    sut.ubar = np.array([4.83,])

    # Initial conditions
    x0 = np.array([4.37, -3.74])

    # Calculate analytical solution
    def get_ref_sol(t):
        npts = t.shape[0]
        x_ref = np.empty((npts, 2))
        x_ref[:, 0] = (0.5 * sut.ubar * t**2) + (x0[1] * t) + x0[0]
        x_ref[:, 1] = x0[1] + sut.ubar * t

        y_ref = x_ref[:, 0].reshape((npts, 1))

        return SysSolution(t, x_ref, y_ref)

    return SystemTestCase(sut, x0, get_ref_sol)

def test_simple_integrator_constant():
    # Setup simple integrator with constant input
    I = integrator.SimpleIntegrator()
    I.ubar = np.array([4.13,])

    # Solution params
    tf = 10
    x0 = 15.2
    npts = 100

    # Reference solution
    t_ref = np.linspace(0, tf, npts)
    x_ref = (x0 + t_ref * I.ubar).reshape((npts, 1))

    sim = I.compute_trajectory(x0, tf=tf, n=npts)

    assert(np.allclose(t_ref, sim.t))
    assert(np.allclose(x_ref, sim.x))
    assert(np.allclose(x_ref, sim.y))

def test_simple_integrator_ramp():
    """Simple integrator with ramp input"""
    I = integrator.SimpleIntegrator()

    # Solution params
    tf = 10
    x0 = 39.4
    npts = 100

    # Ramp input
    ramp_cst = np.pi
    def u(t):
        return np.asarray(t * ramp_cst)

    # Reference solution
    t_ref = np.linspace(0, tf, npts)
    x_ref = (x0 + 0.5 * ramp_cst * t_ref**2).reshape((npts, 1))

    sim = I.compute_trajectory(x0, tf=tf, n=npts, u=u)

    assert(np.allclose(t_ref, sim.t))
    assert(np.allclose(x_ref, sim.x))
    assert(np.allclose(x_ref, sim.y))

def test_double_integrator_constant_ode(double_integ):
    # Solution params
    tf = 10
    npts = 10
    t_ref = np.linspace(0, tf, npts)

    # Reference solution
    ref_sol = double_integ.get_ref_sol(t_ref)

    # Solution computed by SUT
    I = double_integ.sut
    sim = I.compute_trajectory(double_integ.x0, tf=tf, n=npts, solver='ode')

    assert(np.allclose(t_ref, sim.t))
    assert(np.allclose(ref_sol.x, sim.x))
    assert(np.allclose(ref_sol.y, sim.y))

def test_double_integrator_constant_euler(double_integ):
    # Solution params
    tf = 10
    npts = 100
    t_ref = np.linspace(0, tf, npts)

    # Reference solution
    ref_sol = double_integ.get_ref_sol(t_ref)

    # Solution computed by SUT
    I = double_integ.sut
    sim = I.compute_trajectory(double_integ.x0, tf=tf, n=npts, solver='euler')

    # Euler's method has low-order convergence, so we tolerate 1% error
    atol, rtol = 1, 0.01

    assert(np.allclose(t_ref, sim.t))
    assert(np.allclose(ref_sol.x, sim.x, atol=atol, rtol=rtol))
    assert(np.allclose(ref_sol.y, sim.y, atol=atol, rtol=rtol))