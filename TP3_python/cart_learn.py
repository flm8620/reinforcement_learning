import numpy as np
import lqg1d
import matplotlib.pyplot as plt
import utils
from cartpole import CartPoleEnv

class ConstantStep(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update(self, gt):
        return self.learning_rate * gt


class AnnealingStep(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.count = 10

    def update(self, gt):
        self.count += 1
        return self.learning_rate * gt / self.count


class AdamStep(object):
    def __init__(self, learning_rate):
        self.beta1=0.9
        self.beta2=0.999
        self.eps=1e-8
        self.alpha = learning_rate
        self.m_old = 0
        self.v_old = 0
        self.betaT1 = self.beta1
        self.betaT2 = self.beta2


    def update(self, gt):
        mt = self.beta1*self.m_old + (1-self.beta1) * gt
        vt = self.beta2*self.v_old + (1-self.beta2) * (gt*gt)
        m_hat = mt/(1-self.betaT1)
        v_hat = vt/(1-self.betaT2)
        self.betaT1 *= self.beta1
        self.betaT2 *= self.beta2
        return self.alpha * m_hat / (np.sqrt(v_hat)+self.eps)


#####################################################
# Define the environment and the policy
#####################################################
env = CartPoleEnv()


class Policy(object):
    def __init__(self):
        self.theta = np.zeros(8)
        self.K = 1.0

    def draw_action(self, state):
        if np.random.random()<self.pi(0,state):
            return 0
        else:
            return 1

    def pi(self, a, s):
        theta1 = self.theta[0:4]
        theta2 = self.theta[4:8]
        e0 = np.exp(self.K*np.dot(s, theta1))
        e1 = np.exp(self.K*np.dot(s, theta2))
        if a == 0:
            return e0/(e0+e1)
        if a == 1:
            return e1/(e0+e1)

    def gradQ(self, a, s):
        if a == 0:
            return np.array([s[0],s[1],s[2],s[3],0,0,0,0])
        if a == 1:
            return np.array([0,0,0,0,s[0],s[1],s[2],s[3]])


#####################################################
# Experiments parameters
#####################################################
# We will collect N trajectories per iteration
N = 30
# Each trajectory will have at most T time steps
T = 1000
# Number of policy parameters updates
n_itr = 400
# Set the discount factor for the problem
discount = 1.0
# Learning rate for the gradient update
learning_rate = 0.1
# trial
trials = 1


params = 'N='+str(N)+' T='+str(T)
avg_return = np.zeros(n_itr)
for t in range(trials):
    #####################################################
    # define the update rule (stepper)
    schema = 'AdamStep=' + str(learning_rate)
    # stepper = ConstantStep(learning_rate)
    # stepper = AnnealingStep(learning_rate) # e.g., constant, adam or anything you want
    stepper = AdamStep(learning_rate)
    # fill the following part of the code with
    #  - REINFORCE estimate i.e. gradient estimate
    #  - update of policy parameters using the steppers
    #  - average performance per iteration
    #  - distance between optimal mean parameter and the one at it k


    policy = Policy()
    for it in range(n_itr):
        print(it, 'theta=', policy.theta)
        paths = utils.collect_episodes(env, policy=policy, horizon=T, n_episodes=N)

        gradJ = 0.
        total_R = 0.
        # plt.figure()
        # plt.plot(paths[0]['states'])
        # plt.show()
        for k in range(N):
            p = paths[k]
            R_tau = 0.
            gamma_n = 1.
            sum_grad = 0.
            for i in range(len(p['actions'])):
                a = np.asscalar(p['actions'][i])
                s = p['states'][i]
                sum_grad += policy.K*(policy.gradQ(a, s) - policy.pi(0, s)*policy.gradQ(0, s)-policy.pi(1, s)*policy.gradQ(1, s))

                r = np.asscalar(p['rewards'][i])
                R_tau += r*gamma_n
                gamma_n *= discount
            gradJ += sum_grad * R_tau
            total_R += R_tau
        gradJ /= N
        policy.theta += stepper.update(gradJ)
        # policy.theta = min(max(policy.theta, -500), 500)

        avg_return[it] += total_R/N

avg_return /= trials

# plot the average return obtained by simulating the policy
# at each iteration of the algorithm (this is a rought estimate
# of the performance
plt.figure()
plt.plot(avg_return)
name = 'Avg_return '+params+' '+schema
plt.title(name)
# plt.savefig(name.replace(' ','_')+'.eps', format='eps', dpi=800)
plt.show()
# plot the distance mean parameter

utils.collect_episodes(env, policy=policy, horizon=2000, n_episodes=10, render=True)