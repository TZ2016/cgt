import numpy as np


def rmsprop_create(theta, step_size, decay_rate=.95, eps=1.e-6):
    optim_state = dict(theta=theta, step_size=step_size,
                   decay_rate=decay_rate,
                   sqgrad=np.zeros_like(theta) + eps,
                   scratch=np.empty_like(theta), count=0, type='rmsprop')
    return optim_state

def rmsprop_update(grad, state):
    state['sqgrad'][:] *= state['decay_rate']
    state['count'] *= state['decay_rate']
    np.square(grad, out=state['scratch']) # scratch=g^2
    state['sqgrad'] += state['scratch']
    state['count'] += 1
    np.sqrt(state['sqgrad'], out=state['scratch']) # scratch = sum of squares
    np.divide(state['scratch'], np.sqrt(state['count']), out=state['scratch']) # scratch = rms
    np.divide(grad, state['scratch'], out=state['scratch']) # scratch = grad/rms
    np.multiply(state['scratch'], state['step_size'], out=state['scratch'])
    state['theta'][:] += state['scratch']

def adam_create(theta, step_size=1.e-3, beta1=.9, beta2=.999, eps=1.e-8):
    optim_state = dict(theta=theta, step_size=step_size,
                       beta1=beta1, beta2=beta2, eps=eps, _t=0,
                       _m=np.zeros_like(theta), _v=np.zeros_like(theta),
                       scratch=np.zeros_like(theta), type='adam')
    return optim_state

def adam_update(grad, state):
    state['_t'] += 1
    state['_m'] *= state['beta1']
    state['_m'] += (1 - state['beta1']) * grad
    state['_v'] *= state['beta2']
    np.square(grad, out=state['scratch'])
    state['_v'] += (1 - state['beta2']) * state['scratch']
    np.sqrt(state['_v'], out=state['scratch'])
    np.divide(state['_m'], state['scratch'] + state['eps'], out=state['scratch'])
    state['scratch'] *= state['step_size'] * \
                        np.sqrt(1. - state['beta2'] ** state['_t']) / \
                        (1. - state['beta1'] ** state['_t'])
    state['theta'] += state['scratch']
