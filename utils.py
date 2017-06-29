
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def lrelu(x, leak=0.2):
    with tf.name_scope('leak_relu'):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def normc_initializer(std=1.0):
    """
    Initialize array with normalized columns
    """
    def _initializer(shape, dtype=None, partition_info=None): #pylint: disable=W0613
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

def dense(x, size, name, weight_init=None):
    """
    Dense (fully connected) layer
    """
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
    b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
    return tf.matmul(x, w) + b


def clip_by_value(inp, range):
    ret = inp.copy()
    for i in xrange(len(inp)):
        ret[i] = min(inp[i], range[1])
        ret[i] = max(inp[i], range[0])
    return ret

def dense_homogeneous_network(inputs, layer_sizes, hidden_afunc = None, output_afunc = None, output_range = None):
    ''' 
    
    :param inputs: The inputs to the nn
    :param layer_sizes: Output size of each layer. The input size is dictated by the previous layer. 
    :param hidden_afunc: A hidden activation function for all hidden layers
    :param output_afunc: output activation function
    :param output_range: clip to this range or scale to this range. 
    :return: a fully dense neural network with same activation function for all hidden layers and the weights and biases in a python list
    '''
    n_hidden = len(layer_sizes) - 1
    hidden_afunc_list = [hidden_afunc] * n_hidden
    return dense_network(inputs, layer_sizes, hidden_afunc = hidden_afunc_list, output_afunc = output_afunc, output_range = output_range)


def dense_network(inputs, layer_sizes, hidden_afunc = None, output_afunc = None, output_range = None):
    '''
    
    :param inputs: The inputs to the nn
    :param layer_sizes: Output size of each layer. THe input size is dictated by the previous layer. 
    :param hidden_afunc: A list of hidden activation function for each layer. 
    :param output_afunc: output activation function
    :param output_range: clip to this range or scale to this range. 
    :return: a fully dense neural network and the weights and biases in a python list
    '''

    hidden_sizes = layer_sizes[:-1]
    n_hidden = len(hidden_sizes)

    if hidden_afunc is None:
        hidden_afunc = [None] * n_hidden

    w = [0]*len(layer_sizes)
    b = [0]*len(layer_sizes)

    # Build hidden layers
    nn = inputs
    for i in xrange(n_hidden):
        nn, w[i], b[i] = add_layer(nn, hidden_sizes[i], activation_function=hidden_afunc[i])

    nn, w[-1], b[-1] = add_layer(nn, layer_sizes[-1], activation_function=output_afunc, output_range = output_range)
    return nn, w, b

def explained_variance_1d(ypred,y):
    """
    Var[ypred - y] / var[y]. 
    https://www.quora.com/What-is-the-meaning-proportion-of-variance-explained-in-linear-regression
    """
    assert y.ndim == 1 and ypred.ndim == 1
    vary = np.var(y)
    return np.nan if vary==0 else 1 - np.var(y-ypred)/vary


def add_layer(inputs, out_size, activation_function=None, output_range = None):
    # add one more layer and return the output of this layer

    in_size = int(inputs.get_shape()[1])

    if activation_function == tf.nn.tanh:
        r = tf.sqrt(6./(in_size+out_size))
        Weights = tf.Variable(tf.random_uniform([in_size, out_size], minval = -r, maxval = r))
    else:
        r = tf.sqrt(6./(in_size+out_size))
        Weights = tf.Variable(r*tf.random_normal([in_size, out_size])) #TODO change r. Hardcoded here
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    if output_range is None:
        return outputs, Weights, biases

    else:
        b1 = output_range[0]
        b2 = output_range[1]

        if activation_function == tf.nn.tanh:
            a1 = np.array([-1.0] * out_size)
            a2 = np.array([1.0] * out_size)
            return tf.multiply((outputs - a1) , tf.div((b2-b1),(a2-a1))) + b1, Weights, biases

        elif activation_function is None: # TODO clip by value
            return outputs, Weights, biases

        else:
            raise ValueError("Cannot specify output range for this activation function")


def cg(Ax_handle, b, x_init = None, eps = 1e-8, n_iters = float('inf')):
    '''

    :param Ax_handle: Handle to the hessian vector product
    :param b:
    :param x_init:
    :param eps:
    :param n_iters:
    :return:
    '''

    A = Ax_handle
    if x_init is None:
        r = b.copy() # negative gradient
        x = 0
    else:
        r = b - Ax_handle(x_init) # negative gradient
        x = x_init.copy()
    if n_iters is None:
        n_iters = np.shape(b)[0]
    p = r.copy()
    rdotr = r.dot(r)
    iter = 0

    while True:
        iter +=1
        if iter > n_iters or rdotr <= eps:
            break
        Ap= A(p)
        alpha = rdotr/p.dot(Ap)
        r -= alpha * Ap
        x += alpha*p
        new_rdotr = r.dot(r)
        beta = new_rdotr/rdotr
        p = r + beta*p
        rdotr = new_rdotr

    return x





class neural_net: # TODO investigate the best way to have networks and sessions

    def __init__(self):
        pass
    def train(self):
        pass
# Abstracts current fsm state  using automata into a class

class fsm:
    def __init__(self, P):
        '''

        :param P: State transition matrix
        '''
        self.state = None

    def set_state(self, s):
        self.state = s

    def take_input(self, s):
        '''
        Takes in the passed string to compute the new fsm state
        :param s: input string
        '''

        pass