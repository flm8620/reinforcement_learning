import numpy as np
import lqg1d
# from fqi import FQI
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from utils import collect_episodes, estimate_performance


env = lqg1d.LQG1D(initial_state_type='random')
discount = 0.9
horizon = 50
N = 100

actions = discrete_actions = np.linspace(-8, 8, 20)


#################################################################
# Show the optimal Q-function
#################################################################
def make_grid(x, y):
    m = np.meshgrid(x, y, copy=False, indexing='ij')
    return np.vstack(m).reshape(2, -1).T

states = discrete_states = np.linspace(-10, 10, 20)
SA = make_grid(states, actions)
S, A = SA[:, 0], SA[:, 1]

K, cov = env.computeOptimalK(discount), 0.001
print('Optimal K: {} Covariance S: {}'.format(K, cov))

Q_fun_ = np.vectorize(lambda s, a: env.computeQFunction(s, a, K, cov, discount, 1))
Q_fun = lambda X: Q_fun_(X[:, 0], X[:, 1])

Q_opt = Q_fun(SA)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(S, A, Q_opt)


#################################################################
# Collect the samples using the behavioural policy
#################################################################
# You should use discrete actions


class UniformPolicy(object):
    def __init__(self, a):
        self.actions = a

    def draw_action(self, state):
        return np.random.choice(self.actions)


beh_policy = UniformPolicy(discrete_actions)

dataset = collect_episodes(env, n_episodes=N, policy=beh_policy, horizon=horizon)

# define FQI
# to evaluate the policy you can use estimate_performance


class FQI(object):
    def __init__(self):
        self.theta = np.array([0, 0, 0.01, 0.01, 0, 0])
        pass

    def d(self):
        return 6

    def phi(self, s, a):
        return np.array([s, a*s, a*a, s*s, a, np.ones(s.shape)])

    def Q(self, s, a):
        return np.dot(self.theta, self.phi(s, a))

    def argmaxQ(self, s):
        return - (self.theta[0]+s*self.theta[1])/2/self.theta[2]

    def maxQ(self, s):
        return self.Q(s, self.argmaxQ(s))

    def flatten_data(self, dataset, N, horizon):
        a = []
        s = []
        s_next = []
        r = []
        for i in range(N):
            d = dataset[i]
            for t in range(horizon):
                a.append(np.asscalar(d['actions'][t]))
                s.append(np.asscalar(d['states'][t]))
                r.append(np.asscalar(d['rewards'][t]))
                s_next.append(np.asscalar(d['next_states'][t]))
        a = np.array(a)
        s = np.array(s)
        r = np.array(r)
        s_next = np.array(s_next)
        return a, s, r, s_next

    def learn(self, dataset, N, horizon, J):
        lamb = 10
        a, s, r, s_next = self.flatten_data(dataset, N, horizon)
        Z = self.phi(s, a)
        ZZ = np.dot(Z, Z.T)
        for k in range(10):
            yt = r + discount * self.maxQ(s_next)
            A = ZZ + lamb * np.identity(self.d())
            self.theta = np.linalg.solve(A, np.dot(Z, yt))
            print(k, ' theta learnt = ', self.theta)
            J.append(estimate_performance(env, policy=self, horizon=100, n_episodes=500, gamma=discount))
            print('Policy performance: {}'.format(J))

    def draw_action(self, state):
        return self.argmaxQ(state)

fqi = FQI()
J=[]
fqi.learn(dataset, N, horizon, J)

# plot obtained Q-function against the true one
Q_learn = fqi.Q(S, A)
ax.scatter(S, A, Q_learn)
plt.show()

plt.plot(J)
plt.show()
