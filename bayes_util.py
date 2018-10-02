import tensorflow as tf
import numpy as np
import numpy.linalg as nla

pi = tf.constant(np.pi, dtype=tf.float32)
sqrt2 = tf.constant(np.sqrt(2.0), dtype=tf.float32)
twopi = tf.constant(2.0 * np.pi, dtype=tf.float32)
sqrt2pi = tf.constant(np.sqrt(2.0 * np.pi), dtype=tf.float32)
one_ovr_sqrt2pi = tf.constant(1.0 / np.sqrt(2.0 * np.pi), dtype=tf.float32)
one_ovr_sqrt2 = tf.constant(1.0 / np.sqrt(2.0), dtype=tf.float32)
log2pi = tf.constant(np.log(2.0 * np.pi), dtype=tf.float32)


EPSILON = tf.constant(1e-6)
HALF_EPSILON = EPSILON / 2.0

def standard_gaussian(x):
    return one_ovr_sqrt2pi * tf.exp(-x*x / 2.0)

def gaussian_cdf(x):
    return 0.5 * (1.0 + tf.erf(x * one_ovr_sqrt2))

def softrelu(x):
    return standard_gaussian(x) + x * gaussian_cdf(x)

def g(rho, mu1, mu2):
    one_plus_sqrt_one_minus_rho_sqr = (1.0 + tf.sqrt(1.0 - rho*rho))
    a = tf.asin(rho) - rho / one_plus_sqrt_one_minus_rho_sqr
    safe_a = tf.abs(a) + HALF_EPSILON
    safe_rho = tf.abs(rho) + EPSILON
    
    A = a / twopi
    sxx = safe_a * one_plus_sqrt_one_minus_rho_sqr / safe_rho
    one_ovr_sxy = (tf.asin(rho) - rho) / (safe_a * safe_rho)
    
    return A * tf.exp(-(mu1*mu1 + mu2*mu2) / (2.0 * sxx) + one_ovr_sxy * mu1 * mu2)

def delta(rho, mu1, mu2):
    return gaussian_cdf(mu1) * gaussian_cdf(mu2) + g(rho, mu1, mu2)

def heavy_g(rho, mu1, mu2):
    sqrt_one_minus_rho_sqr = tf.sqrt(1.0 - rho*rho)
    a = tf.asin(rho)
    safe_a = tf.abs(a) + HALF_EPSILON
    safe_rho = tf.abs(rho) + EPSILON
    
    A = a / twopi
    sxx = safe_a * sqrt_one_minus_rho_sqr / safe_rho
    sxy = safe_a * sqrt_one_minus_rho_sqr * (1 + sqrt_one_minus_rho_sqr) / (rho * rho)
    return A * tf.exp(-(mu1*mu1 + mu2*mu2) / (2.0 * sxx) + mu1*mu2/sxy)

#DEBUG utils
def make_random_covariance_matrix(dim):
    factor = 2
    tmp = np.random.randn(factor * dim, dim)
    C = np.matmul(tmp.T, tmp)
    return (C / nla.norm(C, 2)).astype(np.float32)


