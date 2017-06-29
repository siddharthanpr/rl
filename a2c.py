# Continuous cart pole using policy gradients (PG)
# Running this script does the trick!


import gym
import numpy as np
import tensorflow as tf
from gym import wrappers
from utils import *
import scipy.signal
import logz


from time import time
# env = gym.make('InvertedPendulum-v1')
# env = wrappers.Monitor(env, '/home/sid/ccp_pg', force= True)




def discount(x, gamma):
    """
    Compute discounted sum of future values
    out[i] = in[i] + gamma * in[i+1] + gamma^2 * in[i+2] + ...
    """

    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]


def surface_plot():
    fig = plt.figure()
    ax = Axes3D(fig)
    X = np.arange(-1, 1, 0.05)
    Y = np.arange(-1, 1, 0.05)
    l = np.shape(X)[0]
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros([l, l])
    inp = np.zeros([1, n])

    for i in xrange(l):
        for j in xrange(l):
            inp[0][0] = X[i][j]
            inp[0][2] = Y[i][j]

            Z[i][j] = sess.run(nn_v, feed_dict={sy_obs_p: inp})

    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    plt.show()


class LinearValueFunction(object):

    coef = None

    def fit(self, X, y, **_):
        Xp = self.preproc(X)
        A = Xp.T.dot(Xp)
        nfeats = Xp.shape[1]
        A[np.arange(nfeats), np.arange(nfeats)] += 1e-3 # a little ridge regression
        b = Xp.T.dot(y)
        print 'linear loss before v-fit', np.mean(np.square(self.predict(X) - y))
        self.coef = np.linalg.solve(A, b)
        print 'linear loss after v-fit', np.mean(np.square(self.predict(X) - y))

    def predict(self, X):
        if self.coef is None:
            return np.zeros(X.shape[0])
        else:
            return self.preproc(X).dot(self.coef)


    def preproc(self, X):
        return np.concatenate([np.ones([X.shape[0], 1]), X, np.square(X)/2.0], axis=1)


class NNValueFunction(object):

    '''
    Builds tf graph. Do not create object after initializing graph.
    '''

    def __init__(self, in_size, hidden_sizes=[16], hidden_afunc=tf.nn.relu, learning_rate = 0.001, reg_lambda = 0.001, optimizer_handle = tf.train.GradientDescentOptimizer, \
                                                                                                                                   n_epochs = 10):
        '''
        initializes a homogeneous
        :param input_p: input place holder
        '''

        self.first_fit = True
        self.feat_means = np.zeros(in_size)
        self.feat_stds = np.ones(in_size)
        self.sess = None
        preproc_size = self.preproc(np.zeros((1, in_size))).shape[1]
        self.sy_target_p = tf.placeholder(tf.float32, [None], name="target")
        self.sy_input_p = tf.placeholder(tf.float32, [None, preproc_size], name = "input")
        self.nn_v, _, _ = dense_homogeneous_network(self.sy_input_p, layer_sizes=hidden_sizes + [1], hidden_afunc=hidden_afunc)  # [1] for output layer because value function is real valued
        self.n_epochs = n_epochs

        l1_regularizer = tf.contrib.layers.l1_regularizer(scale = reg_lambda, scope=None)
        regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, tf.trainable_variables())

        self.sy_loss = tf.reduce_mean(tf.square(self.sy_target_p - self.nn_v)) + regularization_penalty
        optimizer_v = optimizer_handle(learning_rate=learning_rate)
        # gvs_v = optimizer_v.compute_gradients(self.sy_loss_value)
        # # clipped_gvs_v = [(tf.clip_by_value(grad, -10., 10.), var) for grad, var in gvs_v if grad is not None]
        # self.train_op_value = optimizer_v.apply_gradients(gvs_v)
        self.train_op_value = optimizer_v.minimize(self.sy_loss)

    def register_session(self, sess):
        self.sess = sess

    def fit(self, X, y, verbose = True):

        assert self.sess is not None, 'Register session first'

        st = time()

        print 'nn loss before v-fit', self._total_loss(X, y)
        for _ in xrange(self.n_epochs):
            self.sess.run(self.train_op_value, feed_dict = {self.sy_input_p : self.preproc(X),self.sy_target_p : y})

        if verbose:
            print 'V-Fitting time', time() - st
        print 'nn loss after v-fit', self._total_loss(X, y)


    def set_stats(self, X):
        self.feat_means = X.mean(axis = 0)
        self.feat_stds = X.std(axis = 0)

    def normalized(self, X):
        return (X - self.feat_means)/(self.feat_stds+1e-8)

    def _total_loss(self, X, y):
        return self.sess.run(self.sy_loss, feed_dict = {self.sy_target_p : y, self.sy_input_p : self.preproc(X)})

    def predict(self, X):
        assert self.sess is not None, 'Register session first'
        return np.squeeze(self.sess.run(self.nn_v, feed_dict = {self.sy_input_p : self.preproc(X)}))

    def preproc(self, X):
        # return self.normalized(X)

        # return X
        return np.concatenate([X, np.square(X)], axis=1)




def init_session(sess):
    if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
        init = tf.initialize_all_variables()
    else:
        init = tf.global_variables_initializer()
    sess.run(init)
    sess.__enter__()
    return sess

def flatten_view(var):
    '''
    retruns a flatterned view of the variables
    :param var:
    :return:
    '''
    return tf.concat([tf.reshape(i, [-1]) for i in var], axis = 0)


def main(env, desired_kl=2e-3, vf_type='nn', vf_params={}, gamma=0.95, animate=False, max_timesteps_per_batch = 2500, n_iter=300, n_steps = 50,
         initial_stepsize=1e-3):

    sess = tf.Session()
    stepsize = initial_stepsize  # initial step size
    mvnd = tf.contrib.distributions.MultivariateNormalDiag # MultivariateNormalDiagonal distribution
    n = len(env.reset()) # |S|
    k = np.shape(env.action_space)[0]
    action_range = [env.action_space.low.astype(np.float32), env.action_space.high.astype(np.float32)]


    sy_obs_p = tf.placeholder(tf.float32, [None, n], name = "observation")
    sy_ac_p = tf.placeholder(tf.float32, [None, k], name = "action")
    sy_adv_p = tf.placeholder(tf.float32, [None], name = "advantage")

    with tf.name_scope('mean_action_network'):
        nn_ac_mean, _, _ = dense_homogeneous_network(sy_obs_p, layer_sizes=[32, k], hidden_afunc=lrelu, output_afunc=tf.nn.tanh, output_range=action_range)
    #
    # with tf.name_scope('mean_action_network'):
    #     sy_h1 = lrelu(dense(sy_obs_p, 32, "h1", weight_init=normc_initializer(1.0))) # hidden layer 1
    #     # sy_h2 = lrelu(dense(sy_h1, num_dim_2, "h2", weight_init=normc_initializer(1.0)))
    #     nn_ac_mean = dense(sy_h1, k, "final", weight_init=normc_initializer(0.1)) # output layer
    #     tf.summary.histogram('nn_ac_mean', nn_ac_mean)


    sy_log_sigma = tf.Variable(tf.zeros([k,]))
    theta = flatten_view(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='mean_action_network') + [sy_log_sigma])# parameters of the policy

    sy_ac_distr = mvnd(loc=nn_ac_mean, scale_diag=tf.exp(sy_log_sigma))
    sy_sample_action = tf.stop_gradient(sy_ac_distr.sample())

    sy_loss_policy = -tf.reduce_mean(tf.multiply(sy_ac_distr.log_prob(sy_ac_p),sy_adv_p))
    sy_g = tf.gradients(sy_loss_policy, theta)
    sy_mean1_p = tf.placeholder(tf.float32, [None, k], name = "means1")
    sy_mean2_p = tf.placeholder(tf.float32, [None, k], name = "means2")
    sy_log_sigma1_p = tf.placeholder(tf.float32, [None, k], name = "std1")
    sy_log_sigma2_p = tf.placeholder(tf.float32, [None, k], name = "std2")

    sy_kl_trajectories = tf.reduce_mean(tf.contrib.distributions.kl(mvnd(loc=sy_mean1_p, scale_diag=tf.exp(sy_log_sigma1_p)),
                                                                    mvnd(loc=sy_mean2_p, scale_diag=tf.exp(sy_log_sigma2_p)))) # Symbolic

    sy_ky_old_varnew = tf.reduce_mean(tf.contrib.distributions.kl(mvnd(loc=sy_mean1_p, scale_diag=tf.exp(sy_log_sigma1_p)),
                                                                    mvnd(loc=nn_ac_mean, scale_diag=tf.exp(sy_log_sigma)))) # Symbolic

    sy_trpo_surr = sy_loss_policy + 
    sy_ent = k/2.*tf.log(2*np.pi) + k/2. + tf.reduce_sum(sy_log_sigma)

    sy_stepsize_p = tf.placeholder(tf.float32, [], name = "step_size")
    optimizer_p = tf.train.AdamOptimizer(learning_rate = sy_stepsize_p)
    train_op_policy = optimizer_p.minimize(sy_loss_policy)


    if vf_type == 'nn':
        vf_params['in_size'] =n

        vf = NNValueFunction(**vf_params)
        vf. register_session(sess)

    elif vf_type == 'linear':
        vf = LinearValueFunction()
    else:
        raise 'Invalid vf_type'

    # TRPO stuff
    sy_vector_p = tf.placeholder(tf.float32, tf.shape(theta), name = "vector")
    sy_grad = tf.gradients(sy_ky_old_varnew, theta)
    sy_gradient_vector_product = tf.reduce_sum(tf.multiply(sy_grad_p, sy_vector_p))
    sy_hessian_vector_product = tf.gradients(sy_gradient_vector_product_p, theta)


    sess = init_session(sess)




    max_timesteps = 80000000


    T = 0
    my_method = True
    start_time = None
    for i in xrange(n_iter):
        obs_n, returns_n, acs_n, episode_rewards = [], [], [], []

        if start_time is not None:
            print 'time taken = ', time() - start_time
        start_time = time()
        print("********** Iteration %i ************"%i)
        r = 0
        T_batch = 0
        ob = env.reset()
        total_reward = 0


        while True:
            animate_this_episode = (len(obs_n) == 0 and (i % 10 == 0) and animate)
            obs, rews, acs = [], [], []
            for _ in xrange(n_steps):
                if animate_this_episode:
                    env.render()
                obs.append(ob)
                ac = sess.run(sy_sample_action,feed_dict = {sy_obs_p: ob[None]}) # todo separate noise and mean to get more effiecient samples in loop
                ac_clipped = clip_by_value(ac, action_range)
                ob, rew, done, _ = env.step(ac_clipped)
                rews.append(rew)
                acs.append(ac)
                total_reward += rew

                T += 1
                T_batch +=1
                if done:
                    ob = env.reset()
                    episode_rewards.append(total_reward)
                    total_reward = 0
                    break

            if not done:
                # print type(rews[-1]), vf.predict(ob[None]), type(rews[-1] + gamma * vf.predict(ob[None])[0])
                rews[-1] += gamma * vf.predict(ob[None])[0]

            returns = discount(rews, gamma)

            returns_n.append(returns) # O(1)
            obs_n.append(obs)  # O(1)
            acs_n.append(acs)  # O(1)

            if T_batch > max_timesteps_per_batch:
                break

        obs_n = np.concatenate(obs_n)
        acs_n = np.concatenate(acs_n)
        episode_rewards = np.array(episode_rewards)
        returns_n = np.concatenate(returns_n)
        vtarg_n = returns_n
        vpred_n = vf.predict(obs_n)

        advs_n = returns_n - vpred_n
        standardized_advs_n = (advs_n - advs_n.mean()) / (advs_n.std() + 1e-8)

        old_means = sess.run(nn_ac_mean, feed_dict={sy_obs_p: obs_n})
        old_log_sigma = sess.run(sy_log_sigma, feed_dict={sy_obs_p: obs_n})

        g = sess.run(sy_g, feed_dict={sy_ac_p:acs_n, sy_adv_p: standardized_advs_n, sy_obs_p: obs_n})
        def Ax(x):
            return sess.run(sy_hessian_vector_product, feed_dict={sy_vector_p: x, sy_mean1_p: old_means, sy_log_sigma1_p: old_log_sigma[None]})

        su = cg(Ax, g) # unscaled step
        s = np.sqrt(2*desired_kl/ tf.sum(tf.multiply(su, Ax(su)))) * su



        #update parameters
        sess.run(train_op_policy, feed_dict={sy_ac_p:acs_n, sy_adv_p: standardized_advs_n, sy_obs_p: obs_n, sy_stepsize_p: stepsize})
        vf.fit(obs_n, vtarg_n)

        new_means = sess.run(nn_ac_mean, feed_dict={sy_obs_p: obs_n})
        new_log_sigma = sess.run(sy_log_sigma, feed_dict={sy_obs_p: obs_n})

        kl = sess.run(sy_kl_trajectories,feed_dict={sy_mean1_p: old_means, sy_mean2_p: new_means,sy_log_sigma1_p: old_log_sigma[None], sy_log_sigma2_p: new_log_sigma[None]})
        ent = sess.run(sy_ent)

        logz.log_tabular("EpRewMean", episode_rewards.mean())
        logz.log_tabular("KLOldNew", kl)
        logz.log_tabular("Entropy", ent)
        logz.log_tabular("EVBefore", explained_variance_1d(vpred_n, vtarg_n))
        logz.log_tabular("EVAfter", explained_variance_1d(vf.predict(obs_n), vtarg_n))
        logz.log_tabular("TimestepsSoFar", T)
        # If you're overfitting, EVAfter will be way larger than EVBefore.
        # Note that we fit value function AFTER using it to compute the advantage function to avoid introducing bias
        logz.dump_tabular()

        if kl > desired_kl * 2:
            stepsize /= 1.5
            print('stepsize -> %s' % stepsize)
        elif kl < desired_kl / 2:
            stepsize *= 1.5
            print('stepsize -> %s' % stepsize)
        else:
            print('stepsize OK')

        if T > max_timesteps:
            break



if __name__ == "__main__":
    tf.reset_default_graph()

    env = gym.make('Pendulum-v0')
    # env = gym.make('InvertedPendulum-v1')
    logz.configure_output_dir()
    if 0:
        main(env = env, desired_kl=2e-3, vf_type='linear', gamma=.97, animate=False, max_timesteps_per_batch=2500, n_iter=300,
             initial_stepsize=1e-3, n_steps = 2000)
    if 1:
        main(env = env, desired_kl=2e-3, vf_type='nn', gamma=0.97, animate=False, max_timesteps_per_batch=2500, n_iter=300,
             initial_stepsize=1e-3, vf_params=dict(hidden_sizes = [32], reg_lambda = 0.0, hidden_afunc = lrelu, learning_rate = 0.001, optimizer_handle =
             tf.train.AdamOptimizer, n_epochs = 10), n_steps = 2000)  # when you want to start collecting

        # results, set the logdir


print 'Done'
