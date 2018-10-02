import tensorflow as tf
import numpy as np
import gaussian_variables as gv
import bayes_layers as bnn

class MLP(object):
    def __init__(self, in_size, out_size, hypers):
        hidden_dims = []
        if 'hidden_dims' in hypers:
            hidden_dims = hypers['hidden_dims']
        self.hypers = hypers
        self.layer_factory = self.get_layer_factory(hypers['nonlinearity'])
        self.sizes = [in_size] + hidden_dims + [out_size]
        self.make()
    
    def get_layer_factory(self, nonlinearity):
        nonlinearity = nonlinearity.strip().lower()
        if nonlinearity == "relu":
            return bnn.linear_relu
        elif nonlinearity == "heaviside":
            return bnn.linear_heaviside
    
    def make_placeholders(self):
        self.placeholders = {
            'ipt_mean': tf.placeholder(tf.float32, [None, self.sizes[0]]),
        }
    
    def make(self):
        self.A = []; self.b = []
        for in_dim, out_dim in zip(self.sizes[:-1], self.sizes[1:]):
            self.A.append(gv.make_weight_matrix((in_dim, out_dim), self.hypers['prior_type']))
            self.b.append(gv.make_bias_vector((in_dim, out_dim), self.hypers['prior_type']))
        self.parameters = self.A + self.b
    
    def __call__(self, x):
        A = self.A; b = self.b
        h = bnn.linear_certain_activations(x, A[0].value, b[0].value)
        for L in range(1, len(A)):
            h = self.layer_factory(h, A[L].value, b[L].value)
        return h

    def run_with_MC(self, x, n_sample):
        A = [a.value.sample(n_sample) for a in self.A]
        b = [tf.expand_dims(b.value.sample(n_sample), axis=1) for b in self.b]
        
        h = tf.matmul(x, A[0]) + b[0]
        
        for L in range(1, len(A)):
            h = tf.matmul(tf.nn.relu(h), A[L]) + b[L]
        return h
    
    def get_weights(self, sess, pickleable=False):
        fetch_dict = {
            'b_mean' : [b.value.mean for b in self.b],            'b_var'  : [b.value.var  for b in self.b],
            'A_mean' : [A.value.mean for A in self.A],            'A_var'  : [A.value.var  for A in self.A],
            'b_prior_mean' : [b.prior.mean for b in self.b],      'b_prior_var'  : [b.prior.var  for b in self.b],
            'A_prior_mean' : [A.prior.mean for A in self.A],      'A_prior_var'  : [A.prior.var  for A in self.A]
        }
        
        weights_dict = sess.run(fetch_dict)
        
        if pickleable:
            return weights_dict
        else:
            return self.parse_weights_dict(weights_dict)
        
    def parse_weights_dict(self, weights_dict):        
        weights_dict = {
            'b' : [gv.Parameter(gv.DiagonalGaussianVar(m, v), gv.DiagonalGaussianVar(pm, pv))
                    for b,m,v,pm,pv in zip(self.b, weights_dict['b_mean'], weights_dict['b_var'],
                                                   weights_dict['b_prior_mean'], weights_dict['b_prior_var'])],
            'A' : [gv.Parameter(gv.DiagonalGaussianVar(m, v), gv.DiagonalGaussianVar(pm, pv))
                    for A,m,v,pm,pv in zip(self.A, weights_dict['A_mean'], weights_dict['A_var'],
                                                   weights_dict['A_prior_mean'], weights_dict['A_prior_var'])]
        }
        return weights_dict
    
    def set_weights(self, sess, weights_dict):
        weights_dict = self.parse_weights_dict(weights_dict)
        assign_ops = []
        for b, b_param in zip(self.b, weights_dict['b']):
            assign_ops.append(tf.assign(b.value.mean, b_param.value.mean))
            assign_ops.append(tf.assign(b.variables['log_sigma'], 0.5*np.log(b_param.value.var)))
        for A, A_param in zip(self.A, weights_dict['A']):
            assign_ops.append(tf.assign(A.value.mean, A_param.value.mean))
            assign_ops.append(tf.assign(A.variables['log_sigma'], 0.5*np.log(A_param.value.var)))
        sess.run(tf.group(assign_ops))
        
    
    def from_prior(self):
        assigners = []
        for w in self.parameters:
            assigners += [tf.assign(w.value.mean, w.prior.sample()),
                          tf.assign(w.variables['log_sigma'], -11.0*tf.ones(w.value.shape, dtype=tf.float32))]
        return tf.group(*assigners)
        
        
class PointMLP(MLP):    
    def from_MLP(self, mlp):
        self.A = mlp.A; self.b = mlp.b
        self.parameters = self.A + self.b

    def __call__(self, x):
        A = [aa.value.mean for aa in self.A]
        b = [bb.value.mean for bb in self.b]
        self.h = [x]
        self.h.append(tf.matmul(x, A[0]) + b[0])
        for L in range(1, len(A)):
            self.h.append(tf.matmul(tf.nn.relu(self.h[-1]), A[L]) + b[L])
        return gv.GaussianVar(self.h[-1], tf.constant(0.0, dtype=tf.float32))

    
class AdaptedMLP(MLP):
    def __init__(self, mlp):
        self.mlp = mlp
        self.__dict__.update(mlp.__dict__)
        self.make_adapters()

    def make_adapters(self):
        self.adapter = {}
        for ad in ['in', 'out']:
            self.adapter[ad] = {
                'scale': tf.Variable(self.hypers['adapter'][ad]['scale'], trainable=False),
                'shift': tf.Variable(self.hypers['adapter'][ad]['shift'], trainable=False)
            }

    def __call__(self, x):
        x_ad = self.adapter['in']['scale'] * x + self.adapter['in']['shift']
        self.pre_adapt = self.mlp(x_ad)
        mean = self.adapter['out']['scale'] * self.pre_adapt.mean + self.adapter['out']['shift']
        cov  = tf.reshape(self.adapter['out']['scale'], [-1,1]) * tf.reshape(self.adapter['out']['scale'], [1,-1]) * self.pre_adapt.var
        return gv.GaussianVar(mean, cov)

    def run_with_MC(self, x, n_sample):
        x_ad = self.adapter['in']['scale'] * x + self.adapter['in']['shift']
        self.pre_adapt = self.mlp.run_with_MC(x_ad, n_sample)
        mean = self.adapter['out']['scale'] * self.pre_adapt + self.adapter['out']['shift']
        return mean