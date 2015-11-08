import cgt
import core
import numpy as np
import numpy.random as nr

class Distribution(object):
    def lik(self, x, p):
        raise NotImplementedError
    def loglik(self, x, p):
        raise NotImplementedError
    def logprob(self, x, p):
        raise NotImplementedError
    def crossent(self, p, q):
        raise NotImplementedError
    def kl(self, p, q):
        raise NotImplementedError
    def sample(self, p):
        raise NotImplementedError

# serve as a table to store past samples as an empirical distribution
class Empirical(Distribution):
    pass

class _Bernoulli(Distribution):
    """
    Bernoulli: f(k; p) = p^k (1 - p)^(1 - k), k = 0, 1
    """
    def loglik(self, x, p):
        """ Log likelihood of params on dataset x """
        return cgt.sum(self.logprob(x, p))
    def logprob(self, x, p):
        """ Element-wise log prob for each component in x """
        p = core.as_node(p)
        l = x * p + (1 - x) * (1 - p)
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
            return np.array(nr.rand(*p.shape) <= p, dtype="i2")
bernoulli = _Bernoulli()

class _Categorical(Distribution):
    def crossent(self, p, q):
        assert p.ndim==2 and q.ndim==2
        return -(p*cgt.log(q)).sum(axis=1)
    def loglik(self, labels, p):
        return cgt.log(p[cgt.arange(cgt.size(labels,0)),labels])
categorical = _Categorical()

class _DiagonalGaussian(Distribution):
    pass

class Product(Distribution):    
    r"""
    Factored distribution obtained by taking the product of several component distributions
    E.g. suppose we have p0(x), p1(y), p2(z),
    then p3 := ProductDistribution(p1,p2,p3) is a distribution satisfying
    p3(x,y,z) = p0(x)p1(y)p2(z)
    """
    pass
