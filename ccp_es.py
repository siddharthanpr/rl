## Continous cart pole using evolutionary strategy (ES)

import gym
import numpy as np
from gym import wrappers
env = gym.make('InvertedPendulum-v1')
# env = wrappers.Monitor(env, '/home/sid/ccp_pg', force= True)

total_runs = 0
def simulate(policy, steps, graphics = False):
    observation = env.reset()
    R = 0
    for i in xrange(steps):
        if graphics: env.render()
        a = policy(observation)

        observation, reward, done, info = env.step(a)
        R += reward
        if done:
            break
    return R


def approx_policy_eval(policy, n_samples = 1):
    R = 0
    global total_runs
    total_runs += 1
    for _ in xrange(n_samples):
        observation = env.reset()
        for i in xrange(1000):
            a = policy(observation)

            observation, reward, done, info = env.step(a)

            R += reward
            if done:
                break


    return R/float(n_samples)


def es():

    npop = 10     # population size
    sigma = 0.1    # noise standard deviation
    alpha = 0.1  # learning rate
    n = env.observation_space.shape[0]
    w = np.random.randn(n) # initial guess

    max_U = -float('inf')
    for i in range(1000):
      N = np.random.randn(npop, n)
      R = np.zeros(npop)
      for j in range(npop):
        w_try = w + sigma*N[j]
        R[j] = approx_policy_eval(lambda s: (w_try.dot(s)) )


      s = np.std(R)
      b = np.mean(R)

      if b > max_U:
          max_U = b
          max_w = w.copy()

      print b

      if s != 0:
        A = (R - b) / s
      else:
          A = R
      if b > 950: return w
      w += alpha/(npop*sigma) * np.dot(N.T, A)

    print 'max', max_U
    return max_w

w = es()
r = 0
for i in xrange(100):
    r+=simulate(lambda s: (w.dot(s)), 1000, graphics=1)

print 'average_return over 100 trials:', r/100.0
print 'total episodes', total_runs