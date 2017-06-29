import gym
import numpy as np
env = gym.make('CartPole-v1')

def simulate(policy, steps):
    observation = env.reset()
    R = 0
    for i in xrange(steps):
        env.render()
        a = policy(observation)

        observation, reward, done, info = env.step(a)
        R += reward
        if done:
            print 'Done', i
            break
    return R


def approx_policy_eval(policy, n_samples = 1):
    R = 0
    for _ in xrange(n_samples):
        observation = env.reset()
        for i in xrange(600):
            a = policy(observation)

            observation, reward, done, info = env.step(a)

            R += reward
            if done:
                break


    return R/float(n_samples)


def es():

    npop = 50     # population size
    sigma = 0.1    # noise standard deviation
    alpha = 0.1  # learning rate
    n = env.observation_space.shape[0]
    w = np.random.randn(n) # initial guess
    max_U = -float('inf')
    for i in range(300):
      N = np.random.randn(npop, n)
      R = np.zeros(npop)
      for j in range(npop):
        w_try = w + sigma*N[j]
        R[j] = approx_policy_eval(lambda s: (w_try.dot(s)) > 0)
        if R[j] > max_U:
            max_U = R[j]
            max_w = w_try
      print np.mean(R)


      A = (R - np.mean(R)) / np.std(R)
      if np.std(R) == 0:
          A = R
      w += alpha/(npop*sigma) * np.dot(N.T, A)
      if np.mean(R) > 490: return w
    print 'max', max_U
    return w

w = es()
print 'sim_reward =', simulate(lambda s: (w.dot(s)) > 0, 1000)
print 'sim_reward =', simulate(lambda s: (w.dot(s)) > 0, 1000)
print 'sim_reward =', simulate(lambda s: (w.dot(s)) > 0, 1000)
