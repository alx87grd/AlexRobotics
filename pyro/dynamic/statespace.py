import numpy as np

from scipy import linalg

from pyro.dynamic  import ContinuousDynamicSystem
from pyro.analysis import simulation


###############################################################################
class StateSpaceSystem(ContinuousDynamicSystem):
    """Time-invariant state space representation of dynamic system

    f = A x + B u
    h = C x + D u

    Parameters
    ----------
    A, B, C, D : array_like
        The matrices which define the system

    """
    ############################################
    def __init__(self, A, B, C, D):
        
        self.A = A
        self.B = B
        self.C = C
        self.D = D

        self._check_dimensions()

        n = A.shape[1]
        m = B.shape[1]
        p = C.shape[0]
        
        ContinuousDynamicSystem.__init__( self, n, m, p)
        
    ############################################
    def _check_dimensions(self):
        
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("A must be square")

        if self.B.shape[0] != self.A.shape[0]:
            raise ValueError("Number of rows in B does not match A")

        if self.C.shape[1] != self.A.shape[0]:
            raise ValueError("Number of columns in C does not match A")

        if self.D.shape[1] != self.B.shape[1]:
            raise ValueError("Number of columns in D does not match B")

        if self.C.shape[0] != self.D.shape[0]:
            raise ValueError("Number of rows in C does not match D")
    
    #############################################
    def f(self, x, u, t = 0 ):

        dx = np.dot(self.A, x) + np.dot(self.B, u)

        return dx
    
    #############################################
    def h(self, x, u, t = 0 ):
        
        y = np.dot(self.C, x) + np.dot(self.D, u)
        
        return y
    
    
    ############################################
    def compute_eigen_modes(self):
        
        D,V = linalg.eig( self.A )
        
        self.poles = D
        self.modes = V
        
        return (D,V)
    
    ############################################
    def compute_eigen_mode_traj(self, i = 0 ):
        """ 
        Simulation of time evolution of the system on mode i
        ------------------------------------------------
        i : mode index
        """
        
        #Time scaling for the mode
        norm = np.sqrt(self.poles[i].real**2 + self.poles[i].imag**2)
        
        if norm > 0.001:
            tf = 2. / norm * 2 * np.pi + 1
            tf = np.clip(tf , 1 , 30)
        else:
            tf = 5
            
        n  = 2001

        sim = simulation.Simulator(self, tf, n)
        
        sim.x0 = self.modes[:,i].real + self.xbar

        traj   = sim.compute() # save the result in the instance

        return traj
    
    ############################################
    def animate_eigen_mode(self, i = 0 , is_3d = False):
        """ 
        Simulation of time evolution of the system on mode i
        ------------------------------------------------
        i : mode index
        """
        
        # Compute eigen decomposition
        self.compute_eigen_modes()
        
        # Simulate one mode
        traj = self.compute_eigen_mode_traj( i )
        
        # Animate mode
        animator       = self.get_animator()
        
        template = 'Mode %i \n%0.1f+%0.1fj'
        label    = template % (i, self.poles[i].real, self.poles[i].imag)
        
        animator.top_right_label = label
        animator.animate_simulation( traj, 3.0, is_3d)

    
    
    
    

################################################################
def _approx_jacobian(func, xbar, epsilons):
    """ Numerically approximate the jacobian of a function

    Parameters
    ----------
    func : callable
        Function for which to approximate the jacobian. Must accept an array of
        dimension ``n`` and return an array of dimension ``m``.
    xbar : array_like (dimension ``n``)
        Input around which the jacobian will be evaluated.
    epsilons : array_like (dimension ``n``)
        Step size to use for each input when approximating the jacobian

    Returns
    -------
    jac : array_like
        Jacobian matrix with dimensions m x n
    """

    n  = xbar.shape[0]
    ybar = func(xbar)
    m  = ybar.shape[0]

    J = np.zeros((m, n))
    
    for i in range(n):
        # Forward evaluation
        xf    = np.copy(xbar)
        xf[i] = xbar[i] + epsilons[i]
        yf    = func(xf)

        # Backward evaluation
        xb    = np.copy(xbar)
        xb[i] = xbar[i] - epsilons[i]
        yb    = func(xb)
        
        # Slope
        delta = yf - yb

        J[:, i] = delta / (2.0 * epsilons[i])

    return J


#################################################################
def linearize(sys, epsilon_x=0.001, epsilon_u=None):
    """Generate linear state-space model by linearizing any system.

    The system to be linearized is assumed to be time-invariant.

    Parameters
    ----------
    sys : `pyro.dynamic.ContinuousDynamicSystem`
        The system to linearize
    xbar : array_like
        State array arround which the system will be linearized
    epsilon : float
        Step size to use for numerical gradient approximation

    Returns
    -------
    instance of `StateSpaceSystem`

    """
    
    xbar = sys.xbar.astype(float)
    ubar = sys.ubar.astype(float)

    epsilon_x = np.asarray(epsilon_x)

    if epsilon_u is None:
        if epsilon_x.size > 1:
            raise ValueError("If epsilon_u is not provided, epsilon_x must be scalar")
        epsilon_u = epsilon_x

    epsilon_u = np.asarray(epsilon_u)

    if epsilon_u.size == 1:
        epsilon_u = np.ones(sys.m) * epsilon_u

    if epsilon_x.size == 1:
        epsilon_x = np.ones(sys.n) * epsilon_x
        

    def f_x(x):
        return sys.f(x, ubar, 0)

    def f_u(u):
        return sys.f(xbar, u, 0)

    def h_x(x):
        return sys.h(x, ubar, 0)

    def h_u(u):
        return sys.h(xbar, u, 0)

    A = _approx_jacobian(f_x, xbar, epsilon_x)
    B = _approx_jacobian(f_u, ubar, epsilon_u)
    C = _approx_jacobian(h_x, xbar, epsilon_x)
    D = _approx_jacobian(h_u, ubar, epsilon_u)
    
    ss = StateSpaceSystem(A, B, C, D)
    
    #############
    # Labels
    #############
    
    for i in range(sys.n):
        ss.state_label[i]  = 'Delta ' + sys.state_label[i]
    
    ss.state_units  = sys.state_units
    
    for i in range(sys.p):
        ss.output_label[i] = 'Delta ' + sys.output_label[i]
        
    ss.output_units = sys.output_units
    
    for i in range(sys.m):
        ss.input_label[i]  = 'Delta ' + sys.input_label[i]
        
    ss.input_units  = sys.input_units
    
    ss.name = 'Linearized ' + sys.name
    
    #############
    # Graphical
    #############
    
    # New fonction from delta_states to configuration space
    def new_xut2q( x, u, t):
        
        x = x + sys.xbar
        u = u + sys.ubar
        
        return sys.xut2q( x, u, t)
    
    ss.xut2q                     = new_xut2q
    
    # Using the non-linear sys graphical kinematic
    ss.linestyle                = sys.linestyle
    ss.forward_kinematic_domain = sys.forward_kinematic_domain
    ss.forward_kinematic_lines  = sys.forward_kinematic_lines

    return ss


class StateObserver(StateSpaceSystem):
    """Linear time-invariant continuous-time state observer

    f = d(x_est)/dt = A x_est + B u_sys + L(y_sys - y_est)

    Where x_est is the estimate of x, the state vector and y_est is the
    estimate of y, the output vector, based on x_est and u_sys.

    States of observer (n = n_sys):

        x = x_est

    Inputs of observer (m = m_sys + p_sys):

        u = [u_sys; y_sys]

    Outputs of observer (p = n = n_sys):

        y = x = x_est


    Parameters
    ----------
    A : array-like      n x n
        Systems dynamics (state transition) matrix of the observer plant model
    B : array-like      n x m
        Input matrix of the observer plant model
    C : array-like      p x n
        State-Output matrix of the observer plant model
    D : array-like      p x m
        Input-Output matrix of the observer plant model
    L:  array-like      n x p
        Observer gain matrix. n and p refer respectively to the number of states and the
        number of outputs of `sys`.

    """

    def __init__(self, A, B, C, D, L):
        self.A = np.array(A, ndmin=2, dtype=np.float64)
        self.B = np.array(B, ndmin=2, dtype=np.float64)
        self.C = np.array(C, ndmin=2, dtype=np.float64)
        self.D = np.array(D, ndmin=2, dtype=np.float64)
        self.L = np.array(L, ndmin=2, dtype=np.float64)


        self.m_plant = self.B.shape[1]
        self.p_plant = self.C.shape[0]

        # Observer states = Estimated states
        n = self.A.shape[1]

        # Observer inputs = concatenation of y and u of observed system
        m = self.m_plant + self.p_plant

        # Outputs of observer = estimated states
        p = n

        ContinuousDynamicSystem.__init__(self, n, m, p)

        self._check_dimensions()


    def _check_dimensions(self):
        super()._check_dimensions()

        # L must be n x p of sys
        if self.L.shape[0] != self.n or self.L.shape[1] != self.p_plant:
            raise ValueError("Dimensions of gain matrix L do not match system ss.")


    @classmethod
    def from_ss(cls, ss, L):
        """Create a state observer based on an existing state-space system"""
        obs = cls(ss.A, ss.B, ss.C, ss.D, L)

        obs.name = "Observer for " + ss.name

        obs.state_label = [f"Estimated {l}" for l in ss.state_label]
        obs.state_units = ss.state_units

        obs.input_label = ss.input_label + ss.output_label
        obs.input_units = ss.input_units + ss.output_units

        obs.output_label = obs.state_label
        obs.output_units = obs.state_units

        return obs


    @classmethod
    def kalman(cls, A, B, C, D, Q, R, G=None):
        """ Create a state observer by calculating the Kalman gain matrix.

        This method calculates the Kalman gain matrix L_Kalman for the system:

        dx/dt = Ax + Bu + Gw
        y = Cx + Du + v

        Where w and v are normally distributed random vectors with 0 mean and
        covariance matrices Q and V.

        Notes on matrix G:

            - In the case where the noise process w is additive onto the system inputs,
              dx/dt = Ax + B(u + w), we have B(u + w) = Bu + Bw and therefore `G` = `B`.
              This is the default case when `G` is left unspecified or `None`.

            - If the noise process is additive onto the system states,
              dx/dt = Ax + Bu + w, then `G=I` should be passed as an argument, where `I`
              is the identity matrix with the same shape as A (n x n).

        Parameters
        ----------

        A, B, C, D : array-like
            See description in `StateObserver` class.
        Q : array-like      q x q
            Covariance matrix of the process noise w (q x 1)
        R : array-like      p x p
            Covariance matrix of the output noise v (m x 1)
        G : array-like      n x q
            Input matrix for the noise process w. By default (`G=None`), it is assumed
            that the noise process w is additive on the input u, therefore G = B and
            q = m.

        Returns
        ----------

        Instance of `StateObserver` with L, the Kalman gain matrix. A special
        property `P` is set which corresponds to the state estimation
        covariance matrix.

        """
        Q = np.array(Q, ndmin=2, dtype=np.float64)
        R = np.array(R, ndmin=2, dtype=np.float64)

        if G is None:
            G = B
        else:
            G = np.array(G, ndmin=2, dtype=np.float64)

        L = np.zeros([A.shape[0], C.shape[0]]) # temporary
        obs = cls(A, B, C, D, L)

        # Check dimensions of Q, R and G matrices
        if not Q.shape[0] == Q.shape[1]:
            raise ValueError("Q must be square")
        if not R.shape[0] == R.shape[1]:
            raise ValueError("R must be square")
        if not G.shape[0] == A.shape[0]:
            raise ValueError("Shape[0] of G does not match shape of A")
        if not G.shape[1] == Q.shape[0]:
            raise ValueError("Shape[1] of G does not match shape of Q")
        if not R.shape[0] == C.shape[0]:
            raise ValueError("Shape of R must match number of outputs of C")

        P = linalg.solve_continuous_are(a=obs.A.T, b=obs.C.T, q=(G @ Q @ G.T), r=R)
        LT = np.linalg.solve(R.T, (obs.C @ P.T))
        if LT.ndim < 2:
            LT = LT[:, np.newaxis]
        L_kalm = LT.T
        assert L_kalm.shape == obs.L.shape

        obs.L = L_kalm
        obs.P = P # estimate covariance matrix
        return obs


    @classmethod
    def kalman_from_ss(cls, ss, Q, R, G=None):
        """Create a state observer by calculating the Kalman gain matrix.

        See documentation for `kalman(...)`. This method uses the A, B, C, D matrices
        from the system `ss`.

        Returns
        ----------

        Instance of `StateObserver` with L, the Kalman gain matrix. A special
        property `P` is set which corresponds to the state estimation
        covariance matrix.

        """

        kalm_obs = cls.kalman(ss.A, ss.B, ss.C, ss.D, Q, R, G)
        result = cls.from_ss(ss, kalm_obs.L)
        result.P = kalm_obs.P
        return result


    def h(self, x, u, t):
        # Output of observer system is the full vector of estimated states
        return x


    def f(self, x_est, u, t):

        # Check and adjust dimensions

        assert x_est.size == self.n
        x_est = x_est.reshape(self.n)

        assert u.size == self.m
        u = u.reshape(self.m)
        u_plant, y_plant = u[:self.m_plant], u[self.m_plant]

        assert y_plant.size == self.p_plant

        # State observer equations
        y_est = (self.C @ x_est) + (self.D @ u_plant)
        dx_est = (self.A @ x_est) + (self.B @ u_plant) + (self.L @ (y_plant - y_est))
        return dx_est.flatten()


    def __add__(self, sys):
        return ObservedSystem(sys, self)


class ObservedSystem(ContinuousDynamicSystem):
    """Combination of a dynamic system and a state observer

    States of observed system (n = 2 * n_sys):

        x = [x_sys; x_est]

    Inputs of observed system:

        u = u_sys

    Outputs of observed system:

        y = x_est

    """

    def __init__(self, sys, obs):
        self.sys = sys
        self.obs = obs

        n = sys.n * 2
        m = sys.m
        p = sys.n

        if not sys.n == obs.n:
            raise ValueError("Number of states of observer does not match system")

        if not sys.m == obs.m_plant:
            raise ValueError("Number of plant inputs of observer does not match system")

        if not sys.p == obs.p_plant:
            raise ValueError("Number of plant outputs of observer does not match system")

        super().__init__(n, m, p)

        self.x0 = np.concatenate([self.sys.x0, self.obs.x0], axis=0)

        self.state_label = sys.state_label + [f"Estimated {l}" for l in sys.state_label]
        self.state_units = sys.state_units + sys.state_units

        self.input_label = sys.input_label
        self.input_units = sys.input_units

        self.output_label = self.state_label[:sys.n]
        self.output_units = sys.state_units


    def f(self, x, u, t):
        u = np.array(u, ndmin=1).reshape(self.m)
        n_sys = self.sys.n
        x_sys, x_est = x[:n_sys], x[n_sys:]
        assert x_sys.shape == x_est.shape

        dx_sys = self.sys.f(x_sys, u, t)
        y_sys = self.sys.h(x_sys, u, t)

        u_obs = np.concatenate([u, y_sys], axis=0)
        dx_obs = self.obs.f(x_est, u_obs, t)

        return np.concatenate([dx_sys, dx_obs], axis=0)


    def h(self, x, u, t):
        n_sys = self.sys.n
        x_est = x[n_sys:]
        return x_est


    def t2u(self, t):
        return self.sys.t2u(t)



'''
#################################################################
##################          Main                         ########
#################################################################
'''


if __name__ == "__main__":     
    """ MAIN TEST """
    from pyro.dynamic import pendulum
    
    non_linear_sys = pendulum.SinglePendulum()
    non_linear_sys.xbar = np.array([0.,0.])
    
    EPS = 0.001
    
    linearized_sys = linearize( non_linear_sys , EPS )
    
    print('\nA:\n',linearized_sys.A)
    print('\nB:\n',linearized_sys.B)
    print('\nC:\n',linearized_sys.C)
    print('\nD:\n',linearized_sys.D)
    
    # Small oscillations
    non_linear_sys.x0 = np.array([0.1,0])
    linearized_sys.x0 = np.array([0.1,0])
    
    non_linear_sys.compute_trajectory()
    linearized_sys.compute_trajectory()
    
    non_linear_sys.plot_trajectory()
    linearized_sys.plot_trajectory()
    
    # Large oscillations
    non_linear_sys.x0 = np.array([1.8,0])
    linearized_sys.x0 = np.array([1.8,0])
    
    non_linear_sys.compute_trajectory()
    linearized_sys.compute_trajectory()
    
    non_linear_sys.plot_trajectory()
    linearized_sys.plot_trajectory()
    
    
