
import pytest

import numpy as np

from pyro.dynamic.statespace import StateSpaceSystem, StateObserver

from io import StringIO

class TwoDofSs(StateSpaceSystem):
    def __init__(self):
        A = np.array([
            [-0.00054584,  0.0002618 ],
            [ 0.0002618 , -0.00061508]
        ])

        B = np.array([
            [0.00078177, 0.00028403],
            [0.,         0.00041951]]
        )

        C = np.array([[0., 1.]])

        D = np.array([[0., 0.]])

        super().__init__(A, B, C, D)

class Test_Obs_TwoDofSs():
    """
    Test data (expected Kalman gains, sim results, etc) were generated with Matlab
    script "observer_twodofss.m".
    """
    def test_obs_dimcheck(self):
        sys = TwoDofSs()
        with pytest.raises(ValueError):
            L = np.empty([2, 2])
            StateObserver.from_ss(sys, L)

        with pytest.raises(ValueError):
            L = np.empty([4, 1])
            obs = StateObserver.from_ss(sys, L)

        L = np.empty([2, 1])
        obs = StateObserver.from_ss(sys, L)
        return True

    def test_kalman_gain_from_ss(self):
        sys = TwoDofSs()
        Q = np.array([[0.6324,    0.1880], [0.1880,    0.5469]]);
        R = 0.9575;
        kf = StateObserver.kalman_from_ss(sys, Q, R)

        L_matlab = np.array([0.241879988531013, 0.163053708572278]).reshape(2, 1) * 1E-3
        np.testing.assert_array_almost_equal(kf.L, L_matlab)

    def test_kalman_gain_from_ABCD(self):
        sys = TwoDofSs()
        Q = np.array([[0.6324,    0.1880], [0.1880,    0.5469]]);
        R = 0.9575;
        kf = StateObserver.kalman(sys.A, sys.B, sys.C, sys.D, Q, R)

        L_matlab = np.array([0.241879988531013, 0.163053708572278]).reshape(2, 1) * 1E-3
        np.testing.assert_array_almost_equal(kf.L, L_matlab)

    def test_kalman_check_dims(self):
        sys = TwoDofSs()

        with pytest.raises(ValueError):
            Q = np.empty([3, 3])
            R = np.empty(1)
            kf = StateObserver.kalman_from_ss(sys, Q, R)

        with pytest.raises(ValueError):
            Q = np.empty([2, 3])
            R = np.empty(1)
            kf = StateObserver.kalman_from_ss(sys, Q, R)

        with pytest.raises(ValueError):
            Q = np.empty([3, 2])
            R = np.empty(1)
            kf = StateObserver.kalman_from_ss(sys, Q, R)

        with pytest.raises(ValueError):
            Q = np.empty([2, 2])
            R = np.empty([2, 1])
            kf = StateObserver.kalman_from_ss(sys, Q, R)

        Q = np.empty([2, 2])
        R = np.empty(1)
        kf = StateObserver.kalman_from_ss(sys, Q, R)


    def test_sim_observed_sys(self):
        """Simulate ObservedSystem and compare result against matlab"""
        sys = TwoDofSs()
        Q = np.array([[0.6324,    0.1880], [0.1880,    0.5469]]);
        R = 0.9575;
        kf = StateObserver.kalman_from_ss(sys, Q, R)

        tt = np.linspace(0, 10_000, 50)

        def t2u(t):
            if (t % 2000) < 1000: return [100, 20]
            else: return [150, 20]

        sys.t2u = t2u
        sys.x0 = np.array([50, 20])
        kf.x0 = np.array([0, 0])

        osys = kf + sys
        traj = osys.compute_trajectory(tf=10_000, n=50)

        # Matlab generated data
        expected_x_est_txt = """
        0,0
        17.1816933112253,2.67356729335786
        32.6690586962738,5.82859520370164
        46.6546237710204,9.32442222148156
        59.3067122681008,13.0471771432147
        71.5276536671465,16.9072478692853
        89.4100863270359,21.0605927899578
        105.581133740505,25.5464167418339
        120.235163744847,30.2450926073082
        133.54083561347,35.0605541856489
        144.145342049008,39.908533059689
        147.782605315062,44.4764402774289
        151.218496987128,48.6570058723352
        154.457796350395,52.4874078787767
        157.506432493375,56.0004668309786
        162.611520978705,59.2429648674815
        172.620730276353,62.5084875703597
        181.714021581887,65.8486940494296
        189.992743238983,69.2127999649183
        197.544222204844,72.5606793538811
        201.474865866373,75.8310112090093
        200.549189229051,78.7251242885965
        199.838307324667,81.2244129101249
        199.302580326945,83.3864237545624
        198.90932687086,85.2596270743574
        202.325759488412,86.9320759701282
        209.310514157894,88.7129350344944
        215.639957552738,90.6092996685361
        221.389332577766,92.5772594675247
        226.623132747702,94.5816224562129
        226.987428251076,96.5278400916074
        224.255676111288,98.1279656932221
        221.876950861141,99.406279861416
        219.799635130726,100.417827326237
        217.980439792433,101.208381543566
        221.501471929412,101.90611387318
        227.117466778374,102.793385710491
        232.185808843378,103.840649870868
        236.771941112517,105.003439669918
        240.931870147046,106.245677667186
        238.893939424458,107.421823001517
        235.388555299305,108.281100091856
        232.286339967808,108.871642671038
        229.532653928748,109.244248870842
        227.08108262064,109.441075730228
        231.406190089678,109.643118035093
        236.327062319301,110.07665108912
        240.75426253915,110.692120331065
        244.748083523848,111.444937451339
        248.360465903124,112.298568351518
        """.strip()

        expected_x_est = np.loadtxt(StringIO(expected_x_est_txt), delimiter=",")

        # plots for debugging
        #
        # from matplotlib import pyplot as plt
        # plt.figure()
        # plt.plot(traj.t, expected_x_est, "-o");
        # plt.plot(traj.t, traj.y, "-x");
        # plt.plot(traj.t, traj.x[:, :2], color="k");
        # plt.show()

        # Compare against matlab with 0.1 % error
        np.testing.assert_allclose(expected_x_est, traj.y, rtol=1E-3, atol=0.01)
