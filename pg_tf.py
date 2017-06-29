# Continuous cart pole using policy gradients (PG)
# Running this script does the trick!

import gym
import numpy as np
import sys
import tensorflow as tf
from gym import wrappers
from utils import *
import scipy

env = gym.make('InvertedPendulum-v1')
# env = wrappers.Monitor(env, '/home/sid/ccp_pg', force= True)


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


def roll_out(steps, sigma, episodes):
    pi = 0
    V = 0
    s = {}# state as a function of time
    Gt = {}# Accumulated


    for i in xrange(steps):
        a = pi(observation) + sigma * np.random.randn()

        observation, reward, done, info = env.step(a)
        R += reward
        if done:
            break
    return R

def grad_gaussian_policy(w, sigma, gamma):
    '''
    Computes gradient for a one dimensional continuous action sampled from a gaussian
    :param w: parameters w of the mean of the gaussian policy a = w^Ts
    :param sigma: variance of the policy
    :param gamma: discount factor for variance reduction
    :return: reward and the gradient
    '''
    R_tau = 0
    observation = env.reset()
    grad = 0
    d = 1
    for i in xrange(1000):
        a = w.dot(observation) + sigma * np.random.randn()

        # norm(m,s) => Gaussian distribution with mean m and variance s
        # log norm(w^Ts, sigma^2) = log k/sigma - (a-w^Ts)/sigma^2
        # grad_w log norm(w^Ts, sigma^2) = (a-w^Ts).s/sigma**2
        # grad_sigma log norm(w^Ts, sigma^2) = -1/sigma + (a-w^Ts)^2/sigma^3
        # append 4x1 grad_w with grad_sigma
        grad += np.append(1. / sigma**2 * (a - w.dot(observation)) * observation,    ### 4x1
                          -1. / sigma + 1. / sigma**3 * (a - w.dot(observation))**2) ### 1x1
        observation, reward, done, info = env.step(a)

        R_tau += d * (reward)
        d *= gamma
        if done :
            break
    return R_tau, grad

def sample_gaussian(mean=0.0, stddev=1.0):

    """
    Sample from a normal distribution with diagonal covariance matrix
    :param mean: n-dim vector
    :param stdev: n-dim vector
    :return: sample
    """

    return mean + tf.multiply(stddev, tf.random_normal(shape = [1, k]))

def discount(x, gamma):
    """
    Compute discounted sum of future values
    out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
    """
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]


def main():


    observation = env.reset()
    n = len(observation) # |S|
    k = np.shape(env.action_space)[0]

    sy_obs_p = tf.placeholder(tf.float32, [None, n], name = "observation")
    sy_ac_p = tf.placeholder(tf.float32, [None, k], name = "action")
    sy_neg_log_pi_n = tf.placeholder(tf.float32, [None], name = "action_prob") # symbolic negative log probabilities of taking each action along a trajectory
    sy_adv = tf.placeholder(tf.float32, [None], name = "advantage")

    # add hidden layer
    # l1, _, _ = add_layer(xs, 1, h, activation_function=tf.nn.relu)

    # add output layer
    nn_ac, _, _ = add_layer(sy_obs, n,k)
    sy_log_sigma = tf.Variable(tf.zeros([1, k]))
    sy_sampled_action = nn_ac + tf.multiply(tf.exp(sy_log_sigma), tf.random_normal(shape = [1, k]))

    sy_sigma_inv = tf.div(1.0, tf.exp(sy_log_sigma))
    sy_neg_log_pi_sampled_action =  tf.sum(sy_log_sigma) + 0.5 * tf.matmul((sy_ac_p - nn_ac), tf.multiply(sy_ac_p - nn_ac, sy_sigma_inv), transpose_a = True) #Negative lof prob  of picking the sampled action. tf.sum(sy_log_sigma) because sum of logs = log of products. The products here is the determinant


    loss = tf.reduce_sum()


    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)


    print sess.run(sy_sampled_action, feed_dict={sy_ac:[[0]]})

    n_episodes = 10
    steps = 10
    sess.__enter__()

    gamma = 0.95

    for _ in xrange(n_episodes):
        obs, rews, terminated, acs = [], [], False, []
        ob = env.reset()


        while True:
            for i in xrange(steps):
                ac = sess.run(sy_sampled_action,feed_dict={sy_obs_p:obs})
                neg_log_pi_sampled_action = sess.run(sy_neg_log_pi_sampled_action, feed_dict={sy_ac_p:ac})
                ob, rew, done, info = env.step(ac)
                obs.append(ob)
                rews.append(rew)
                acs.append(ac)


                if done:
                    terminated = True
                    break

            discount(rews, gamma)

    sys.exit()

    # xs = tf.placeholder(tf.float32, [None, 1])
    # ys = tf.placeholder(tf.float32, [None, 1])



main()
w = pg()

print 'Done'
r = 0
for i in xrange(100):
    r += simulate(lambda s: (w.dot(s)), 1000, graphics=1)

print 'average_return over 100 trials:', r/100.0