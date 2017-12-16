import numpy as np
import lqg1d
import matplotlib.pyplot as plt
import utils

class ConstantStep(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, gt):
        return self.learning_rate * gt

#####################################################
# Define the environment and the policy
#####################################################
env = lqg1d.LQG1D(initial_state_type='random')

class Policy(object):
    def __init__(self, param):
        self.theta = param
        self.sigma = 0.5

    def draw_action(self, state):
        return np.random.randn() * self.sigma + self.theta * state

policy = Policy(0.1)

#####################################################
# Experiments parameters
#####################################################
# We will collect N trajectories per iteration
N = 100
# Each trajectory will have at most T time steps
T = 100
# Number of policy parameters updates
n_itr = 100
# Set the discount factor for the problem
discount = 0.9
# Learning rate for the gradient update
learning_rate = 0.0001


#####################################################
# define the update rule (stepper)
stepper = ConstantStep(learning_rate) # e.g., constant, adam or anything you want

# fill the following part of the code with
#  - REINFORCE estimate i.e. gradient estimate
#  - update of policy parameters using the steppers
#  - average performance per iteration
#  - distance between optimal mean parameter and the one at it k
mean_parameters = []
avg_return = []
for it in range(n_itr):
    print(it)
    paths = utils.collect_episodes(env, policy=policy, horizon=T, n_episodes=N)

    gradJ = 0.
    total_R = 0.
    for k in range(N):
        p = paths[k]
        R_tau = 0.
        gamma_n = 1.
        sum_grad = 0.
        for i in range(T):
            a = p['actions'][i]
            s = p['states'][i]
            sum_grad += (a-policy.theta*s)/policy.sigma**2 * s

            r = p['rewards'][i]
            R_tau += r*gamma_n
            gamma_n *= discount
        gradJ += sum_grad * R_tau
        total_R += R_tau
    gradJ /= N
    policy.theta += stepper.update(gradJ)
    policy.theta = min(max(policy.theta, -500), 500)

    avg_return.append(total_R/N)
    mean_parameters.append(policy.theta)


# plot the average return obtained by simulating the policy
# at each iteration of the algorithm (this is a rought estimate
# of the performance
plt.figure()
plt.plot(avg_return)
plt.show()
# plot the distance mean parameter
# of iteration k
plt.figure()
plt.plot(mean_parameters)
plt.show()