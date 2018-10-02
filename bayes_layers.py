import tensorflow as tf
import numpy as np
import utils as u
import bayes_util as bu
import gaussian_variables as gv

EPSILON = 1e-6


def linear(x, A, b):
    """
    compute y = x^T A + b
    """
    x_mean = x.mean
    y_mean = tf.matmul(x_mean, A.mean) + b.mean
    x_cov = x.var
    y_cov = linear_covariance(x_mean, x_cov, A, b)
    return gv.GaussianVar(y_mean, y_cov)

def linear_certain_activations(x_certain, A, b):
    """
    compute y = x^T A + b
    assuming x has zero variance
    """
    x_mean = x_certain
    xx = x_mean*x_mean
    y_mean = tf.matmul(x_mean, A.mean) + b.mean
    y_cov = tf.matrix_diag(tf.matmul(xx, A.var) + b.var)
    return gv.GaussianVar(y_mean, y_cov)

def linear_relu(x, A, b):
    """
    compute y = relu(x)^T A + b
    """
    x_var_diag = tf.matrix_diag_part(x.var)
    sqrt_x_var_diag = tf.sqrt(x_var_diag)
    mu = x.mean / (sqrt_x_var_diag + EPSILON)
    
    def relu_covariance(x):
        mu1 = tf.expand_dims(mu, 2)
        mu2 = tf.transpose(mu1, [0,2,1])

        s11s22 = tf.expand_dims(x_var_diag, axis=2) * tf.expand_dims(x_var_diag, axis=1)
        rho = x.var / (tf.sqrt(s11s22))# + EPSILON)
        rho = tf.clip_by_value(rho, -1/(1+EPSILON), 1/(1+EPSILON))

        return x.var * bu.delta(rho, mu1, mu2)   
    
    z_mean = sqrt_x_var_diag * bu.softrelu(mu)
    y_mean = tf.matmul(z_mean, A.mean) + b.mean
    z_cov = relu_covariance(x)
    y_cov = linear_covariance(z_mean, z_cov, A, b)
    return gv.GaussianVar(y_mean, y_cov)

def linear_relu_diagonal(x, A, b):
    """
    compute y = relu(x)^T A + b
    """
    x_var_diag = x.var
    sqrt_x_var_diag = tf.sqrt(x_var_diag)
    mu = x.mean / (sqrt_x_var_diag + EPSILON)
    
    pdf = bu.standard_gaussian(mu) 
    cdf = bu.gaussian_cdf(mu)
    softrelu = pdf + mu*cdf
    
    z_mean = sqrt_x_var_diag * softrelu
    y_mean = tf.matmul(z_mean, A.mean) + b.mean
    z_var = x_var_diag * (cdf + mu*softrelu - tf.square(softrelu))
    y_cov = linear_covariance_diagonal(z_mean, z_var, A, b)
    return gv.GaussianVar(y_mean, y_cov)

def simple(x, A, b):
    mu = x.mean
    y_mean = tf.matmul(mu, A.mean) + b.mean    
    y_cov = x.var
    return gv.GaussianVar(y_mean, y_cov)


def linear_heaviside(x, A, b):
    """
    compute y = heaviside(x)^T A + b
    """
    x_var_diag = tf.matrix_diag_part(x.var)
    mu = x.mean / (tf.sqrt(x_var_diag) + EPSILON)
    
    def heaviside_covariance(x):
        mu1 = tf.expand_dims(mu, 2)
        mu2 = tf.transpose(mu1, [0,2,1])

        s11s22 = tf.expand_dims(x_var_diag, axis=2) * tf.expand_dims(x_var_diag, axis=1)
        rho = x.var / (tf.sqrt(s11s22))# + EPSILON)
        rho = tf.clip_by_value(rho, -1/(1+EPSILON), 1/(1+EPSILON))

        return bu.heavy_g(rho, mu1, mu2)
    
    z_mean = bu.gaussian_cdf(mu)
    y_mean = tf.matmul(z_mean, A.mean) + b.mean
    z_cov = heaviside_covariance(x)
    y_cov = linear_covariance(z_mean, z_cov, A, b)
    return gv.GaussianVar(y_mean, y_cov)


def linear_covariance_diagonal(x_mean, x_var, A, b):
    xx_mean = x_var + x_mean * x_mean
    term1_diag = tf.matmul(xx_mean, A.var)
    Asqr = tf.square(A.mean)
    A_xCov_A = tf.matmul(x_var, Asqr)
    term2_diag = A_xCov_A
    term3_diag = b.var
    result_diag = term1_diag + term2_diag + term3_diag
    return result_diag

def linear_covariance(x_mean, x_cov, A, b):
    x_var_diag = tf.matrix_diag_part(x_cov)
    xx_mean = x_var_diag + x_mean * x_mean
    
    term1_diag = tf.matmul(xx_mean, A.var)
    
    flat_xCov = tf.reshape(x_cov, [-1, A.shape[0]]) # [b*x, x]
    xCov_A = tf.matmul(flat_xCov, A.mean) # [b*x, y]
    xCov_A = tf.reshape(xCov_A, [-1, A.shape[0], A.shape[1]]) # [b, x, y]
    xCov_A = tf.transpose(xCov_A, [0, 2, 1]) # [b, y, x]
    xCov_A = tf.reshape(xCov_A, [-1, A.shape[0]]) # [b*y, x]
    A_xCov_A = tf.matmul(xCov_A, A.mean) # [b*y, y]
    A_xCov_A = tf.reshape(A_xCov_A, [-1, A.shape[1], A.shape[1]]) # [b, y, y]

    term2 = A_xCov_A
    term2_diag = tf.matrix_diag_part(term2)
    
    term3_diag = b.var
    
    result_diag = term1_diag + term2_diag + term3_diag
    return tf.matrix_set_diag(term2, result_diag)      

def logsumexp(y, keepdims=False):
    """
    compute <logsumexp(y)>
    """
    lse = tf.reduce_logsumexp(y.mean, axis=-1, keep_dims=keepdims)   # [b, 1]
    p = tf.exp(y.mean - lse)  # softmax                              # [b, y]
    pTDiagVar = tf.reduce_sum(p * tf.matrix_diag_part(y.var), axis=-1, keep_dims=keepdims)        # [b, 1]
    pTVarp = tf.squeeze(tf.matmul(tf.expand_dims(p, 1), tf.matmul(y.var, tf.expand_dims(p, 2))), axis=-1) # [b]
    return lse + 0.5 * (pTDiagVar - pTVarp)

def logsoftmax(y):
    """
    compute <logsoftmax(y)>
    """
    return y.mean - logsumexp(y, keepdims=True) # [b, y]

def categorical_loss(logits, target, model, hypers, global_step, MC_samples=-1):
    """
    compute <p(D|w)>_q - lambda KL(q || p)
    """
    lsm = tf.cond(tf.greater(MC_samples, 0),
        lambda: sampled_logsoftmax(logits, MC_samples), # we evaluate the logsoftmax using MC sampling
        lambda: logsoftmax(logits)                      # we evaluate the logsoftmax using the delta approx
    )

    all_surprise = tf.reduce_sum(tf.stack([w.surprise() for w in model.parameters]))
    logprob = tf.reduce_sum(target * lsm, axis=1)
    batch_logprob = tf.reduce_mean(logprob)
    
    lmda = hypers['lambda']
    
    L = lmda * all_surprise / hypers['dataset_size'] - batch_logprob
    return L, batch_logprob, all_surprise

def heteroskedastic_gaussian_loglikelihood(pred, target, global_step, hypers):
    log_variance = tf.reshape(pred.mean[:,1], [-1])
    mean = tf.reshape(pred.mean[:,0], [-1])
    if hypers['method'].lower().strip() == 'bayes':
        sll = tf.reshape(pred.var[:,1,1], [-1])
        smm = tf.reshape(pred.var[:,0,0], [-1])
        sml = tf.reshape(pred.var[:,0,1], [-1])
    else:
        sll = smm = sml = tf.constant(0.0, dtype=tf.float32)
    return gaussian_loglikelihood_core(target, mean, log_variance, smm, sml, sll)

def homoskedastic_gaussian_loglikelihood(pred, target, global_step, hypers):
    log_variance = tf.constant(hypers["homo_logvar_scale"], dtype=tf.float32)
    mean = tf.reshape(pred.mean[:,0], [-1])
    sll = tf.constant(0.0, dtype=tf.float32)
    sml = tf.constant(0.0, dtype=tf.float32)
    if hypers['method'].lower().strip() == 'bayes':
        smm = tf.reshape(pred.var[:,0,0], [-1])
    else:
        smm = tf.constant(0.0, dtype=tf.float32)
    return gaussian_loglikelihood_core(target, mean, log_variance, smm, sml, sll)

def gaussian_loglikelihood_core(target, mean, log_variance, smm, sml, sll):
    return -0.5 * (
        bu.log2pi 
        + log_variance
        + tf.exp(-log_variance + 0.5*sll)
          * (smm + (mean - sml - target)**2)
    )

def regression_loss(pred, target, model, hypers, global_step):
    all_surprise = tf.reduce_sum(tf.stack([w.surprise(hypers, global_step) for w in model.parameters]))
    gaussian_loglikelihood = (     heteroskedastic_gaussian_loglikelihood 
                              if   hypers['style'] == 'heteroskedastic' 
                              else homoskedastic_gaussian_loglikelihood)
    log_likelihood = gaussian_loglikelihood(pred, target, global_step, hypers)
    batch_log_likelihood = tf.reduce_mean(log_likelihood)
    lmda = u.piecewise_anneal(hypers, 'lambda', global_step)
    L = lmda * all_surprise / hypers['dataset_size'] - batch_log_likelihood
    return L, batch_log_likelihood, all_surprise

def point_catagorical_loss(logits, target, model, hypers, global_step, MC_samples=-1):
    logprob = -tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=target))
    return -logprob, logprob, tf.constant(0.0, dtype=tf.float32)

def point_regression_loss(pred, target, model, hypers, global_step):
    gaussian_loglikelihood = (     heteroskedastic_gaussian_loglikelihood 
                          if   hypers['style'] == 'heteroskedastic' 
                          else homoskedastic_gaussian_loglikelihood)
    log_likelihood = gaussian_loglikelihood(pred, target, global_step, hypers)
    batch_log_likelihood = tf.reduce_mean(log_likelihood)
    if hypers['method'].lower().strip() == 'map':
        all_LL = tf.reduce_sum(tf.stack([w.log_likelihood() for w in model.parameters]))
        lmda = u.piecewise_anneal(hypers, 'lambda', global_step)
        L = -lmda * all_LL / hypers['dataset_size'] - batch_log_likelihood
    else:
        all_LL = tf.constant(0)
        L = -batch_log_likelihood
    return L, batch_log_likelihood, all_LL

def sample_activations(acts, n_sample):
    """
    take n_sample samples from acts
    input: acts: GaussianVar [batch_size (b), hidden size (h)]
    """
    sigma_sqr = acts.var                                          # [b, h, h]
    sigma = tf.transpose(tf.cholesky(sigma_sqr), [0,2,1])         # [b, h, h]
    standard_samples = tf.random_normal(
        [tf.shape(sigma)[0], n_sample, tf.shape(sigma)[-1]])      # [b, n_sample, h]
    samples = tf.matmul(standard_samples, sigma) + tf.expand_dims(acts.mean, 1) # [b, n_sample, h]
    return samples

def sampled_logsoftmax(logits, n_sample):
    samples = sample_activations(logits, n_sample)         # [b, n_sample, h]
    softmax_samples = tf.nn.softmax(samples, dim=-1)       # [b, n_sample, h]
    mean_softmax = tf.reduce_mean(softmax_samples, axis=1) # [b, h]
    return tf.log(mean_softmax)