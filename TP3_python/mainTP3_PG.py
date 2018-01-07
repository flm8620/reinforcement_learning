import numpy as np
import lqg1d
import matplotlib.pyplot as plt
import utils


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
env = lqg1d.LQG1D(initial_state_type='random')


class Policy(object):
    def __init__(self, param):
        self.theta = param
        self.sigma = 0.5

    def draw_action(self, state):
        return np.random.randn() * self.sigma + self.theta * state



#####################################################
# Experiments parameters
#####################################################
# We will collect N trajectories per iteration
N = 20
# Each trajectory will have at most T time steps
T = 100
# Number of policy parameters updates
n_itr = 500
# Set the discount factor for the problem
discount = 0.9
# Learning rate for the gradient update
learning_rate = 0.0005
# trial
trials = 5


params = 'N='+str(N)+' T='+str(T)
mean_parameters = np.zeros(n_itr)
avg_return = np.zeros(n_itr)
for t in range(trials):
    #####################################################
    # define the update rule (stepper)
    schema = 'AnnealingStep=' + str(learning_rate)
    # stepper = ConstantStep(learning_rate)
    stepper = AnnealingStep(learning_rate) # e.g., constant, adam or anything you want
    # stepper = AdamStep(learning_rate)
    # fill the following part of the code with
    #  - REINFORCE estimate i.e. gradient estimate
    #  - update of policy parameters using the steppers
    #  - average performance per iteration
    #  - distance between optimal mean parameter and the one at it k


    policy = Policy(-0.1)
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
            for i in range(T):
                a = np.asscalar(p['actions'][i])
                s = np.asscalar(p['states'][i])
                sum_grad += (a-policy.theta*s)/policy.sigma**2 * s

                r = np.asscalar(p['rewards'][i])
                R_tau += r*gamma_n
                gamma_n *= discount
            gradJ += sum_grad * R_tau
            total_R += R_tau
        gradJ /= N
        policy.theta += stepper.update(gradJ)
        policy.theta = min(max(policy.theta, -500), 500)

        avg_return[it] += total_R/N
        mean_parameters[it] += policy.theta

avg_return /= trials
mean_parameters /= trials

# plot the average return obtained by simulating the policy
# at each iteration of the algorithm (this is a rought estimate
# of the performance
plt.figure()
plt.plot(avg_return)
name = 'Avg_return '+params+' '+schema
plt.title(name)
plt.savefig(name.replace(' ','_')+'.eps', format='eps', dpi=800)
plt.show()
# plot the distance mean parameter
# of iteration k
plt.figure()
plt.plot(mean_parameters)
name = 'Theta '+params+' '+schema
plt.title(name)
plt.savefig(name.replace(' ','_')+'.eps', format='eps', dpi=800)
plt.show()