import cgt
import core
import numpy as np
import numpy.random as nr

class Distribution(object):
    def lik(self, x, p):
        raise NotImplementedError
    def loglik(self, x, *params):
        return cgt.sum(self.logprob(x, *params))
    def logprob(self, x, *params):
        raise NotImplementedError
    def crossent(self, p, q):
        raise NotImplementedError
    def kl(self, p, q):
        raise NotImplementedError
    def sample(self, *params):
        raise NotImplementedError

# serve as a table to store past samples as an empirical distribution
class Empirical(Distribution):
    pass

class _Bernoulli(Distribution):
    """
    Bernoulli: f(k; p) = p^k (1 - p)^(1 - k), k = 0, 1
    """
    def logprob(self, x, p):
        """ Element-wise log prob for each component in x """
        p = core.as_node(p)
        # TODO_TZ p can be almost 0 or 1. Below is not a good solution!
        # l = cgt.safe_mul(x, cgt.log(p)) + cgt.safe_mul(1 - x, cgt.log(1. - p))
        l = x * cgt.log(p + 1.e-10) + (1 - x) * cgt.log(1. - p + 1.e-10)
        return l
    def sample(self, p, shape=None, numeric=False):
        """ Element-wise sampling for each component of p """
        # TODO_TZ  maybe cgt has mechanism to eval an expr
        if not numeric:
            p = core.as_node(p)
            shape = shape or cgt.shape(p)
            return cgt.rand(*shape) <= p
        else:
            assert isinstance(p, np.ndarray)
            return np.array(nr.rand(*p.shape) <= p)
bernoulli = _Bernoulli()

class _DiagonalGaussian(Distribution):
    def logprob(self, x, mu, sigma):
        """ Calculate logprob for each row of x, mu, sigma """
        assert sigma.ndim == mu.ndim == x.ndim == 2
        k = x.shape[1]
        log_det = cgt.sum(cgt.log(sigma), axis=1, keepdims=True)
        prob_z = -.5 * (k * np.log(2. * np.pi) + log_det)
        prob_e = cgt.sum(-.5 * ((x - mu) ** 2) / sigma, axis=1, keepdims=True)
        # output shape: (size_batch, 1)
        return prob_z + prob_e
gaussian_diagonal = _DiagonalGaussian()

class _IsotropicGaussian(Distribution):
    def logprob(self, x, mu, sigma):
        """ Calculate logprob for each row of x, mu, sigma """
        assert sigma.ndim == mu.ndim == x.ndim == 2
        k = x.shape[1]
        prob_z = -.5 * (k * np.log(2. * np.pi) + k * cgt.log(sigma))
        prob_e = cgt.sum(-.5 * cgt.broadcast('/', (x - mu) ** 2, sigma, 'xx,x1'),
                         axis=1, keepdims=True)
        # output shape: (size_batch, 1)
        return prob_z + prob_e
gaussian_isotropic = _IsotropicGaussian()

class _Categorical(Distribution):
    def crossent(self, p, q):
        assert p.ndim==2 and q.ndim==2
        return -(p*cgt.log(q)).sum(axis=1)
    def loglik(self, labels, p):
        return cgt.log(p[cgt.arange(cgt.size(labels,0)),labels])
categorical = _Categorical()

class Product(Distribution):    
    r"""
    Factored distribution obtained by taking the product of several component distributions
    E.g. suppose we have p0(x), p1(y), p2(z),
    then p3 := ProductDistribution(p1,p2,p3) is a distribution satisfying
    p3(x,y,z) = p0(x)p1(y)p2(z)
    """
    pass
