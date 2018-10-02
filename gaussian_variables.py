import tensorflow as tf
import numpy as np
import utils as u
import bayes_util as bu

pi = bu.pi

class GaussianVar(object):
    def __init__(self, mean, var, shape=None):
        self.mean = mean
        self.var = var
        self.shape = mean.shape if shape is None else shape

class DiagonalGaussianVar(object):
    def __init__(self, mean, var, shape=None):
        self.mean = mean
        self.var = var
        self.shape = tf.shape(mean) if shape is None else shape
    
    def sample(self, n_sample=None):
        no_sample_dim = False
        if n_sample is None:
            no_sample_dim = True
            n_sample = 1
        s = tf.random_normal(shape=tf.concat([[n_sample], self.shape], axis=0)) * tf.sqrt(self.var) + self.mean
        if no_sample_dim:
            return s[0,...]
        else:
            return s

    def log_likelihood(self, x):
        return -0.5 * (bu.log2pi + tf.log(self.var) + (x - self.mean)**2 / self.var)

class DiagonalLaplaceVar(object):
    def __init__(self, mean, var, shape=None):
        self.mean = mean
        self.var = var
        self.b = tf.sqrt(var / 2)
        self.shape = tf.shape(mean) if shape is None else shape
        
    def sample(self):
        return tf.contrib.distributions.Laplace(loc=self.mean, scale=self.b).sample()
    
    def log_likelihood(self, x):
        return -(tf.log(2*self.b) + tf.abs(x - self.mean)/self.b)
    
class InverseGammaVar(object):
    def __init__(self, alpha, beta, shape=None):
        self.mean = beta / (alpha - 1.0)
        self.var = self.mean* self.mean / (alpha - 2.0)
        self.alpha = alpha
        self.beta = beta
        self.shape = tf.shape(self.mean) if shape is None else shape
        
    
def KL(p, q, hypers=None, global_step=1.0E99):
    if isinstance(p, DiagonalGaussianVar):
        if isinstance(q, DiagonalGaussianVar):
            safe_qvar = q.var + bu.EPSILON
            entropy_term = 0.5 * (1 + bu.log2pi + tf.log(p.var))
            cross_entropy_term = 0.5 * (bu.log2pi + tf.log(safe_qvar) + (p.var + (p.mean - q.mean)**2) / safe_qvar)           
            return tf.reduce_sum(cross_entropy_term - entropy_term)
        elif isinstance(q, DiagonalLaplaceVar):
            sigma = tf.sqrt(p.var)
            mu_ovr_sigma = p.mean / sigma
            tmp = 2 * bu.standard_gaussian(mu_ovr_sigma) + mu_ovr_sigma * tf.erf(mu_ovr_sigma * bu.one_ovr_sqrt2)
            tmp *= sigma / q.b
            tmp += 0.5 * tf.log(2 * q.b * q.b / (pi * p.var)) - 0.5
            return tf.reduce_sum(tmp)
        elif isinstance(q, InverseGammaVar):
            return EBKL(p, q, hypers, global_step)
    print('unsupported KL')
    
def EBKL(p, q, hypers=None, global_step=1.0E99):
    if isinstance(p, DiagonalGaussianVar):
        if isinstance(q, InverseGammaVar):
            m = tf.to_float(tf.reduce_prod(tf.shape(p.mean)))
            S = tf.reduce_sum(p.var + p.mean*p.mean)
            m_plus_2alpha_plus_2 = m + 2.0*q.alpha + 2.0
            S_plus_2beta = S + 2.0*q.beta
            
            tmp = m * tf.log(S_plus_2beta / m_plus_2alpha_plus_2)
            tmp += S * (m_plus_2alpha_plus_2 / S_plus_2beta)
            tmp += -(m + tf.reduce_sum(tf.log(p.var)))
            return 0.5 * tmp
    print('unsupported KL')
            

class Parameter(object):
    def __init__(self, value, prior, variables=None):
        self.value = value
        self.prior = prior
        self.variables = variables
    
    def surprise(self, hypers=None, global_step=1.0E99):
        """
        compute KL(value || prior) 
        assuming
            (1) diagonal gaussian value
            (2) diagonal gaussian prior with scalar mu, var
        """
        return KL(self.value, self.prior, hypers, global_step)

    def log_likelihood(self):
        return tf.reduce_sum(self.prior.log_likelihood(self.value.mean))

    def standardize(self):
        return DiagonalGaussianVar(
            (self.value.mean - self.prior.mean) / np.sqrt(self.prior.var),
            self.value.var / self.prior.var)
    
def glorot_init(shape):
    init_range = np.sqrt(6.0 / (shape[-2] + shape[-1]))
    return np.random.uniform(-init_range, init_range, size=shape).astype(np.float32)

def gaussian_init(mean, sigma, shape):
    return mean + sigma * np.random.randn(*shape).astype(np.float32)

def laplace_init(mean, sigma, shape):
    return np.random.laplace(mean, sigma / np.sqrt(2.0), size=shape).astype(np.float32)

def get_variance_scale(initialization_type, shape):
    if initialization_type == "standard":
        prior_var = 1.0
    elif initialization_type == "wide":
        prior_var = 100.0
    elif initialization_type == "narrow":
        prior_var = 0.01
    elif initialization_type == "glorot":
        prior_var = (2.0 / (shape[-1] + shape[-2]))
    elif initialization_type == "xavier":
        prior_var = 1.0/shape[-2]
    elif initialization_type == "he":
        prior_var = 2.0/shape[-2]
    elif initialization_type == "wider_he":
        prior_var = 5.0/shape[-2]
    else:
        raise NotImplementedError('prior type "%s" not recognized' % initialization_type)
    return prior_var

def make_weight_matrix(shape, prior_type):
    s2 = get_variance_scale(prior_type[1].strip().lower(), shape)
    sigma = np.sqrt(np.ones(shape) * s2).astype(np.float32)
    log_sigma = np.log(sigma)
    log_sigma = tf.Variable(log_sigma)
    sigma = tf.exp(log_sigma)

    if prior_type[0].strip().lower() == 'gaussian':
        init_function = gaussian_init
        prior_generator = DiagonalGaussianVar
    elif prior_type[0].strip().lower() == 'laplace':
        init_function = laplace_init
        prior_generator = DiagonalLaplaceVar
    elif prior_type[0].strip().lower() == 'empirical':
        a = 4.4798
        alpha = tf.Variable(a, dtype=tf.float32, trainable=False)
        beta = tf.Variable((1+a) * s2, dtype=tf.float32, trainable=False)
        
        mean = tf.Variable(gaussian_init(0.0, np.sqrt(s2), shape))
        value = DiagonalGaussianVar(mean, sigma*sigma, shape)
        prior = InverseGammaVar(alpha, beta)
        
        return Parameter(value, prior, {'mean': mean, 'log_sigma': log_sigma})
    else:
        raise NotImplementedError('prior type "%s" not recognized' % prior_type[0])
    
    mean = init_function(0.0, np.sqrt(s2), shape)
    mean = tf.Variable(mean)
    value = DiagonalGaussianVar(mean, sigma*sigma, shape)
    
    prior_mean = tf.Variable(np.broadcast_to(0.0, shape), dtype=tf.float32, trainable=False)
    prior_var  = tf.Variable(np.broadcast_to(s2,  shape), dtype=tf.float32, trainable=False)
    prior = prior_generator(prior_mean, prior_var, shape)
    
    return Parameter(value, prior, {'mean': mean, 'log_sigma': log_sigma})

def make_bias_vector(shape, prior_type):    
    fudge_factor = 10.0
    s2 = get_variance_scale(prior_type[2].strip().lower(), shape)
    sigma = np.ones((shape[-1],)).astype(np.float32) * np.sqrt(s2 / fudge_factor)
    log_sigma = np.log(sigma)
    log_sigma = tf.Variable(log_sigma)
    sigma = tf.exp(log_sigma)
    
    if prior_type[0].strip().lower() == 'gaussian':
        prior_generator = DiagonalGaussianVar
    elif prior_type[0].strip().lower() == 'laplace':
        prior_generator = DiagonalLaplaceVar
    elif prior_type[0].strip().lower() == 'empirical':
        a = 4.4798
        alpha = tf.Variable(a, dtype=tf.float32, trainable=False)
        beta = tf.Variable((a + 1.0) * s2, dtype=tf.float32, trainable=False)
        
        mean = tf.Variable(np.zeros((shape[-1],)).astype(np.float32))
        value = DiagonalGaussianVar(mean, sigma*sigma, (shape[-1],))
        prior = InverseGammaVar(alpha, beta)
        
        return Parameter(value, prior, {'mean': mean, 'log_sigma': log_sigma})
    else:
        raise NotImplementedError('bias prior type "%s" not recognized' % prior_type[0])
    
    mean = np.zeros((shape[-1],)).astype(np.float32)
    mean = tf.Variable(mean)
    value = DiagonalGaussianVar(mean, sigma*sigma, (shape[-1],))
    
    prior_mean = tf.Variable(np.broadcast_to(0.0, (shape[-1],)), dtype=tf.float32, trainable=False)
    prior_var  = tf.Variable(np.broadcast_to(s2,  (shape[-1],)), dtype=tf.float32, trainable=False)
    prior = prior_generator(prior_mean, prior_var, (shape[-1],))
    
    return Parameter(value, prior, {'mean': mean, 'log_sigma': log_sigma})