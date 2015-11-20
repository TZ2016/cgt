# Learning Stochastic Feedforward Neural Networks

import os
import cgt
import time
import warnings
import traceback
import pprint
import matplotlib.pyplot as plt
from cgt.core import get_surrogate_func
from cgt import nn
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from scipy.special import expit as sigmoid
from param_collection import ParamCollection
from cgt.distributions import gaussian_diagonal
from demo_char_rnn import make_rmsprop_state


DUMP_ROOT = os.path.join(os.environ['HOME'], 'workspace/cgt/tmp/')
DEFAULT_ARGS = {
    # network architecture
    "num_inputs": 2,
    "num_outputs": 1,
    "num_units": [2, 4, 4],
    "num_sto": [0, 2, 2],
    "no_bias": False,
    "const_var": .05,

    # training parameters
    "n_epochs": 20,
    "step_size": .05,
    "decay_rate": .95,

    # training logistics
    "size_sample": 30,  # #times to sample the network per data pair
    "size_batch": 1,  # #data pairs for each gradient estimate
    # "init_conf": nn.IIDGaussian(std=.1),
    "init_conf": nn.XavierNormal(scale=1.),
    # "param_penal_wt": 1.e-4,
    # "snapshot": os.path.join(DUMP_ROOT, '_1447739075/__snapshot.pkl'),

    # debugging
    "dbg_batch": 100,
    "dbg_plot_samples": True,
    "dump_path": os.path.join(DUMP_ROOT,'_%d/' % int(time.time())),
}


def err_handler(type, flag):
    print type, flag
    traceback.print_stack()
    raise FloatingPointError('refer to err_handler for more details')
np.seterr(divide='call', over='warn', invalid='call', under='warn')
np.seterrcall(err_handler)
np.set_printoptions(precision=4, suppress=True)
print cgt.get_config(True)


def scale_data(Xs, scalars=None):
    if not scalars:
        scalars = [StandardScaler() for _ in range(len(Xs))]
    assert len(scalars) == len(Xs)
    Xs = [scalar.fit_transform(X) for X, scalar in zip(Xs, scalars)]
    return Xs

def generate_examples(N, x, y, p_y):
    X = x * np.ones((N, x.size))
    Y = y * np.ones((N, y.size))
    for i, p in enumerate(p_y):
        if p is not None:
            Y[:, i] = 0.
            Y[:, i][:int(N*p)] = 1.
    np.random.shuffle(Y)
    return X, Y

def data_synthetic_a(N):
    # x = y + 0.3 sin(2 * pi * y) + e, e ~ Unif(-0.1, 0.1)
    Y = np.random.uniform(0., 1., N)
    X = Y + 0.3 * np.sin(2. * Y * np.pi) + np.random.uniform(-.1, .1, N)
    Y, X = Y.reshape((N, 1)), X.reshape((N, 1))
    return X, Y

def data_sigm_multi(N, p):
    if not isinstance(p, (list, tuple)):
        assert isinstance(p, int)
        p = [1. / p] * p
    X = np.random.uniform(-10., 10., N)
    Y = sigmoid(X)
    Y += np.random.normal(0., .1, N)
    y = np.random.multinomial(1, p, size=N)
    Y += np.nonzero(y)[1]
    Y, X = Y.reshape((N, 1)), X.reshape((N, 1))
    return X, Y


def hybrid_layer(X, size_in, size_out, size_random, dbg_out=[]):
    assert size_out >= size_random >= 0
    out = cgt.sigmoid(nn.Affine(
        size_in, size_out, name="InnerProd(%d->%d)" % (size_in, size_out)
    )(X))
    dbg_out.append(out)
    if size_random == 0:
        return out
    if size_random == size_out:
        out_s = cgt.bernoulli(out)
        return out_s
    out_s = cgt.bernoulli(out[:, :size_random])
    out = cgt.concatenate([out_s, out[:, size_random:]], axis=1)
    return out


def hybrid_network(size_in, size_out, num_units, num_stos, dbg_out=[]):
    assert len(num_units) == len(num_stos)
    X = cgt.matrix("X", fixed_shape=(None, size_in))
    prev_num_units, prev_out = size_in, X
    dbg_out.append(X)
    for (curr_num_units, curr_num_sto) in zip(num_units, num_stos):
        _layer_dbg_out = []
        prev_out = hybrid_layer(
            prev_out, prev_num_units, curr_num_units, curr_num_sto,
            dbg_out=_layer_dbg_out
        )
        prev_num_units = curr_num_units
        dbg_out.extend(_layer_dbg_out)
        dbg_out.append(prev_out)
    net_out = nn.Affine(prev_num_units, size_out,
                        name="InnerProd(%d->%d)" % (prev_num_units, size_out)
                        )(prev_out)
    dbg_out.append(net_out)
    return X, net_out


def make_funcs(net_in, net_out, config, dbg_out=None):
    def f_sample(_inputs, num_samples=1, flatten=False):
        _mean, _var = f_step(_inputs)
        _samples = []
        for _m, _v in zip(_mean, _var):
            _s = np.random.multivariate_normal(_m, np.diag(np.sqrt(_v)), num_samples)
            if flatten: _samples.extend(_s)
            else: _samples.append(_s)
        return np.array(_samples)
    Y = cgt.matrix("Y")
    params = nn.get_parameters(net_out)
    size_batch, size_out = net_out.shape
    if 'no_bias' in config and config['no_bias']:
        print "Excluding bias"
        params = [p for p in params if not p.name.endswith(".b")]
    if 'const_var' in config:
        print "Constant variance"
        out_mean = net_out
        out_var = cgt.fill(config['const_var'], [size_batch, size_out])
    else:  # net outputs variance
        cutoff = size_out // 2
        out_mean, out_var = net_out[:, :cutoff], net_out[:, cutoff:]
        # out_var = out_var ** 2 + 1.e-6
        out_var = cgt.exp(out_var) + 1.e-6
    net_out = [out_mean, out_var]
    loss_raw = gaussian_diagonal.logprob(Y, out_mean, out_var)
    if 'param_penal_wt' in config:
        print "Applying penalty on parameter norm"
        assert config['param_penal_wt'] > 0
        params_flat = cgt.concatenate([p.flatten() for p in params])
        loss_param = cgt.fill(cgt.sum(params_flat ** 2), [size_batch, 1])
        loss_param *= config['param_penal_wt']
        loss_raw += loss_param
    # end of loss definition
    f_step = cgt.function([net_in], net_out)
    f_surr = get_surrogate_func([net_in, Y],
                                net_out + dbg_out,
                                [loss_raw], params)
    return params, f_step, None, None, f_surr


def rmsprop_update(grad, state):
    state.sqgrad[:] *= state.decay_rate
    state.count *= state.decay_rate
    np.square(grad, out=state.scratch) # scratch=g^2
    state.sqgrad += state.scratch
    state.count += 1
    np.sqrt(state.sqgrad, out=state.scratch) # scratch = sum of squares
    np.divide(state.scratch, np.sqrt(state.count), out=state.scratch) # scratch = rms
    np.divide(grad, state.scratch, out=state.scratch) # scratch = grad/rms
    np.multiply(state.scratch, state.step_size, out=state.scratch)
    state.theta[:] += state.scratch
    # TODO_TZ double check "+=" is the only change


def train(args, X, Y, dbg_iter=None, dbg_done=None):
    dbg_out = []
    net_in, net_out = hybrid_network(args['num_inputs'], args['num_outputs'],
                                     args['num_units'], args['num_sto'],
                                     dbg_out=dbg_out)
    params, f_step, f_sample, _, f_surr = \
        make_funcs(net_in, net_out, args, dbg_out=dbg_out)
    param_col = ParamCollection(params)
    if 'snapshot' in args:
        print "Loading params from previous snapshot"
        optim_state = pickle.load(open(args['snapshot'], 'r'))
    else:
        optim_state = make_rmsprop_state(theta=param_col.get_value_flat(),
                                         step_size=args['step_size'],
                                         decay_rate=args['decay_rate'])
        optim_state.theta = nn.init_array(
            args['init_conf'], (param_col.get_total_size(), 1)).flatten()
    param_col.set_value_flat(optim_state.theta)
    print "Initialization"
    pprint.pprint(param_col.get_values())
    num_epochs, num_iters = 0, 0
    while num_epochs < args['n_epochs']:
        ind = np.random.choice(X.shape[0], args['size_batch'])
        x, y = X[ind], Y[ind]  # not sure this works for multi-dim
        info = f_surr(x, y, num_samples=args['size_sample'])
        grad = info['grad']
        rmsprop_update(param_col.flatten_values(grad), optim_state)
        param_col.set_value_flat(optim_state.theta)
        num_iters += 1
        if num_iters == Y.shape[0]:
            num_epochs += 1
            num_iters = 0
        if dbg_iter:
            dbg_iter(num_epochs, num_iters, info, param_col, optim_state,
                     f_step, f_surr)
    if dbg_done: dbg_done(param_col, optim_state)
    return optim_state


def example_debug(args, X, Y):
    def safe_path(rel_path):
        abs_path = os.path.join(out_path, rel_path)
        d = os.path.dirname(abs_path)
        if not os.path.exists(d):
            warnings.warn("Making new directory: %s" % d)
            os.makedirs(d)
        return abs_path
    N, _ = X.shape
    out_path = args['dump_path']
    conv_smoother = lambda x: np.convolve(x, [1. / N] * N, mode='valid')
    # cache
    ep_net_distr = []
    it_loss_surr = []
    it_theta_comp = []
    it_grad_norm, it_grad_norm_comp = [], []
    it_theta_norm, it_theta_norm_comp = [], []
    def dbg_iter(num_epochs, num_iters, info, param_col, optim_state, f_step, f_surr):
        it_loss_surr.append(info['objective'])
        it_grad_norm.append(np.linalg.norm(optim_state.scratch))
        it_grad_norm_comp.append([np.linalg.norm(g) for g in info['grad']])
        it_theta_norm.append(np.linalg.norm(optim_state.theta))
        it_theta_norm_comp.append([np.linalg.norm(t)
                                   for t in param_col.get_values()])
        it_theta_comp.append(np.copy(optim_state.theta))
        if num_iters == 0:  # new epoch
            print "Epoch %d" % num_epochs
            print "network parameters"
            pprint.pprint(param_col.get_values())
            print "Gradient norm = %f" % it_grad_norm[-1]
            if args['dbg_plot_samples']:
                s_X = np.random.choice(X.flatten(), size=(args['dbg_batch'], 1), replace=False)
                s_Y_mean, s_Y_var = f_step(s_X)
                err_plt = lambda: plt.errorbar(s_X.flatten(), s_Y_mean.flatten(),
                                               yerr=np.sqrt(s_Y_var).flatten(), fmt='none')
                ep_net_distr.append((num_epochs, err_plt))
    def dbg_done(param_col, optim_state):
        # save params
        pickle.dump(args, open(safe_path('args.pkl'), 'w'))
        pickle.dump(param_col.get_values(), open(safe_path('params.pkl'), 'w'))
        pickle.dump(optim_state, open(safe_path('__snapshot.pkl'), 'w'))
        pickle.dump(np.array(it_theta_comp), open(safe_path('params_history.pkl'), 'w'))
        plt.close('all')
        # plot overview
        plt.figure(); plt.suptitle('overview')
        plt.subplot(311); plt.plot(conv_smoother(it_loss_surr)); plt.title('loss')
        plt.subplot(312); plt.plot(conv_smoother(it_grad_norm)); plt.title('grad')
        plt.subplot(313); plt.plot(conv_smoother(it_theta_norm)); plt.title('theta')
        plt.savefig(safe_path('overview.png')); plt.close()
        # plot grad norm component-wise
        _grad_norm_cmp = np.array(it_grad_norm_comp).T
        plt.figure(); plt.suptitle('grad norm layer-wise')
        _num = len(it_grad_norm_comp[0])
        for _i in range(_num):
            plt.subplot(_num, 1, _i+1); plt.plot(conv_smoother(_grad_norm_cmp[_i]))
        plt.savefig(safe_path('norm_grad_cmp.png')); plt.close()
        # plot theta norm component-wise
        _theta_norm_cmp = np.array(it_theta_norm_comp).T
        plt.figure(); plt.suptitle('theta norm layer-wise')
        _num = len(it_theta_norm_comp[0])
        for _i in range(_num):
            plt.subplot(_num, 1, _i+1); plt.plot(conv_smoother(_theta_norm_cmp[_i]))
        plt.savefig(safe_path('norm_theta_cmp.png')); plt.close()
        # plot samples for each epoch
        if args['dbg_plot_samples']:
            for _e, _distr in enumerate(ep_net_distr):
                _ttl = 'epoch_%d.png' % _e
                plt.scatter(X, Y, alpha=0.5, color='y', marker='*')
                plt.gca().set_autoscale_on(False)
                _distr[1](); plt.title(_ttl)
                plt.savefig(safe_path('_sample/' + _ttl)); plt.cla()
    return {'dbg_iter': dbg_iter, 'dbg_done': dbg_done}

if __name__ == "__main__":
    X, Y = data_synthetic_a(1000)
    X, Y = scale_data((X, Y))
    state = train(DEFAULT_ARGS, X, Y,
                  **example_debug(DEFAULT_ARGS, X, Y))

    # X, Y = generate_examples(10, np.array([3.]), np.array([0.]), [.1])
    # X1, Y1 = generate_examples(10,
    #                            np.array([3.]), np.array([0]),
    #                            [0.5])
    # X2, Y2 = generate_examples(10,
    #                            np.array([2.]), np.array([0]),
    #                            [0.1])
    # X3, Y3 = generate_examples(10,
    #                            np.array([4.]), np.array([0]),
    #                            [0.9])
    # X = np.concatenate([X1, X2, X3])
    # Y = np.concatenate([Y1, Y2, Y3])
