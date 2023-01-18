% Generate reference data for StateObserver unit tests
% Tested with Matlab R2022a

A = [[-0.00054584 0.0002618 ]; [ 0.0002618 -0.00061508]];

B = [[0.00078177, 0.00028403]; [0.,         0.00041951]];
G = B;

C = [[0., 1.]];

D = [[0., 0.]];
H = D;

sys = ss(A, [B G], C, [D H]);
sys.InputName = {'u1', 'u2', 'w1', 'w2'};
sys.StateName = {'x1', 'x2'};
sys.OutputName = {'y1'};

Q = [[0.6324    0.1880]; [0.1880    0.5469]];
R = 0.9575;
N = zeros(2, 1);

[kalmf,L,P] = kalman(sys, Q, R, N);
L

%% Simulation

% time
tf = 10000;
t = linspace(0, tf, 1000);

% inputs

u = zeros(size(t, 2), 2);
u(:, 2) = 20; % ambient temps
u((mod(t, 2000) < 1000), 1) = 100;
u((u(:, 1) == 0), 1) = 150;

% noise data
w = zeros(length(t), 2);
%w(:, 1) = 0 * sin(t * 9.648);
%w(:, 1) = 0 * sin(t * 15.76);

osys =  connect(sys, kalmf,{'u1', 'u2', 'w1', 'w2'}, {'x1_e','x2_e'});

x0 = [50 20];
x0_e = [0 0];

[x_est, tOut, stateOut] = lsim(osys, [u w], t, [x0 x0_e]);

close all; figure(); hold on;
plot(t, stateOut);

tq = linspace(0, tf, 50);
x_est_q = interp1(t, x_est, tq);
plot(tq, x_est_q, 'o');

writematrix(x_est_q, 'x_est.csv')


