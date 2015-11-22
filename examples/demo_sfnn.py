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
from cgt.utility.param_collection import ParamCollection
from cgt.distributions import gaussian_diagonal


DUMP_ROOT = os.path.join(os.environ['HOME'], 'workspace/cgt/tmp/')
DEFAULT_ARGS = {
    # network architecture
    "num_inputs": 1,  # size of net input
    "num_outputs": 1,  # size of net output
    "num_units": [2, 4, 4],  # (hidden only) layer-wise number of neurons
    "num_sto": [0, 2, 2],  # (hidden only) layer-wise number of stochastic neurons
    "no_bias": False,  # no bias terms
    "out_var": False,  # net outputs diagonal variance
    "const_var": .05,  # assume isotropic variance in all directions
    "var_in": True,  # variance fed as input

    # training parameters
    "n_epochs": 20,
    "step_size": .05,  # RMSProp,
    "decay_rate": .95,  # RMSProp

    # training logistics
    "size_sample": 30,  # #times to sample the network per data pair
    "size_batch": 1,  # #data pairs for each gradient estimate
    # "init_conf": nn.IIDGaussian(std=.1),
    "init_conf": nn.XavierNormal(scale=1.),  # initialization
    "param_penal_wt": 0.,  # weight decay (0 means none)
    # "snapshot": os.path.join(DUMP_ROOT, '_1447739075/__snapshot.pkl'),

    # debugging
    "debug": True,
    "dump_path": os.path.join(DUMP_ROOT,'_%d/' % int(time.time())),
    "dbg_out_full": True,
    "dbg_plot_samples": True,
    "dbg_batch": 100,
    "dbg_plot_x_dim": 0,
    "dbg_plot_y_dim": 0,
}


def err_handler(type, flag):
    print type, flag
    traceback.print_stack()
    raise FloatingPointError('refer to err_handler for more details')
np.seterr(divide='call', over='warn', invalid='call', under='warn')
np.seterrcall(err_handler)
np.set_printoptions(precision=4, suppress=True)
print cgt.get_config(True)
cgt.check_source()


def scale_data(X, Y, Y_var=None, scalers=None):
    if not scalers:
        scalers = [StandardScaler() for _ in range(2)]
    assert len(scalers) == 2
    s_X = scalers[0].fit_transform(X)
    s_Y = scalers[1].fit_transform(Y)
    if Y_var is not None:
        scalers[1].with_mean = False
        s_Y_var = np.square(scalers[1].transform(np.sqrt(Y_var)))
        return s_X, s_Y, s_Y_var
    return s_X, s_Y

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
    Z = np.random.normal(scale=.05, size=N)
    # X = np.hstack([X, Z])
    # Y = np.hstack([Y, Z])
    i_s =np.argsort(X, axis=0)
    Y, X = np.reshape(Y[i_s], (N, 1)), np.reshape(X[i_s], (N, 1))
    Y_var = np.array([np.var(Y[i:i+20]) for i in range(N-20)] +
                     [np.var(Y[N-20:])] * 20).reshape((N, 1))
    return X, Y, Y_var

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
    net_in = cgt.matrix("X", fixed_shape=(None, size_in))
    prev_num_units, prev_out = size_in, net_in
    dbg_out.append(net_in)
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
    return net_in, net_out


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
    inputs = [net_in]
    if config['no_bias']:
        print "Excluding bias"
        params = [p for p in params if not p.name.endswith(".b")]
    if config['var_in']:
        print "Input includes diagonal variance"
        assert not config['out_var']
        # TODO_TZ diagonal for now
        in_var = cgt.matrix('V', fixed_shape=(None, config['num_inputs']))
        inputs.append(in_var)
        out_mean, out_var = net_out, in_var
    elif config['out_var']:  # net outputs variance
        print "Network outputs diagonal variance"
        cutoff = size_out // 2
        out_mean, out_var = net_out[:, :cutoff], net_out[:, cutoff:]
        # out_var = out_var ** 2 + 1.e-6
        out_var = cgt.exp(out_var) + 1.e-6
    else:
        print "Constant variance"
        out_mean = net_out
        out_var = cgt.fill(config['const_var'], [size_batch, size_out])
    net_out = [out_mean, out_var]
    loss_raw = gaussian_diagonal.logprob(Y, out_mean, out_var)
    if config['param_penal_wt'] != 0.:
        print "Applying penalty on parameter norm"
        assert config['param_penal_wt'] > 0
        params_flat = cgt.concatenate([p.flatten() for p in params])
        loss_param = cgt.fill(cgt.sum(params_flat ** 2), [size_batch, 1])
        loss_param *= config['param_penal_wt']
        loss_raw += loss_param
    # end of loss definition
    f_step = cgt.function(inputs, net_out)
    f_surr = get_surrogate_func(inputs + [Y],
                                net_out + dbg_out,
                                [loss_raw], params)
    # TODO_TZ f_step seems not to fail if X has wrong dim
    return params, f_step, None, None, f_surr


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


def step(X, Y, workspace, config, Y_var=None, dbg_iter=None, dbg_done=None):
    if config['debug'] and (dbg_iter is None or dbg_done is None):
        dbg_iter, dbg_done = example_debug(config, X, Y, Y_var=Y_var)
    f_surr, f_step = workspace['f_surr'], workspace['f_step']
    param_col = workspace['param_col']
    optim_state = workspace['optim_state']
    num_epochs = num_iters = 0
    while num_epochs < config['n_epochs']:
        ind = np.random.choice(X.shape[0], config['size_batch'])
        x, y = X[ind], Y[ind]
        if config['var_in']:
            y_var = Y_var[ind]
            info = f_surr(x, y_var, y, num_samples=config['size_sample'])
        else:
            info = f_surr(x, y, num_samples=config['size_sample'])
        grad = info['grad']
        rmsprop_update(param_col.flatten_values(grad), optim_state)
        param_col.set_value_flat(optim_state['theta'])
        num_iters += 1
        if num_iters == Y.shape[0]:
            num_epochs += 1
            num_iters = 0
        if dbg_iter:
            dbg_iter(num_epochs, num_iters, info, param_col, optim_state,
                     f_step, f_surr)
    if dbg_done: dbg_done(param_col, optim_state)
    return param_col, optim_state


def create(args):
    dbg_out = []
    net_in, net_out = hybrid_network(args['num_inputs'], args['num_outputs'],
                                     args['num_units'], args['num_sto'],
                                     dbg_out=dbg_out)
    if args['dbg_out_full']: dbg_out = []
    params, f_step, f_loss, f_grad, f_surr = \
        make_funcs(net_in, net_out, args, dbg_out=dbg_out)
    param_col = ParamCollection(params)
    if 'snapshot' in args:
        print "Loading params from previous snapshot"
        optim_state = pickle.load(open(args['snapshot'], 'r'))
        assert isinstance(optim_state, dict)
    else:
        theta = param_col.get_value_flat()
        optim_state = dict(theta=theta, step_size=args['step_size'],
                           decay_rate=args['decay_rate'],
                           sqgrad=np.zeros_like(theta) + 1.e-6,
                           scratch=np.empty_like(theta), count=0)
        optim_state['theta'] = nn.init_array(
            args['init_conf'], (param_col.get_total_size(), 1)).flatten()
    param_col.set_value_flat(optim_state['theta'])
    print "Initialization"
    pprint.pprint(param_col.get_values())
    workspace = {
        'optim_state': optim_state,
        'param_col': param_col,
        'f_surr': f_surr,
        'f_step': f_step,
        'f_loss': f_loss,
        'f_grad': f_grad
    }
    return workspace


def example_debug(args, X, Y, Y_var=None):
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
    _ix, _iy = args['dbg_plot_x_dim'], args['dbg_plot_y_dim']
    # cache
    ep_net_distr = []
    it_loss_surr = []
    it_theta_comp = []
    it_grad_norm, it_grad_norm_comp = [], []
    it_theta_norm, it_theta_norm_comp = [], []
    def dbg_iter(num_epochs, num_iters, info, param_col, optim_state, f_step, f_surr):
        it_loss_surr.append(info['objective'])
        it_grad_norm.append(np.linalg.norm(optim_state['scratch']))
        it_grad_norm_comp.append([np.linalg.norm(g) for g in info['grad']])
        it_theta_norm.append(np.linalg.norm(optim_state['theta']))
        it_theta_norm_comp.append([np.linalg.norm(t)
                                   for t in param_col.get_values()])
        it_theta_comp.append(np.copy(optim_state['theta']))
        if num_iters == 0:  # new epoch
            print "Epoch %d" % num_epochs
            print "network parameters"
            pprint.pprint(param_col.get_values())
            print "Gradient norm = %f" % it_grad_norm[-1]
            if args['dbg_plot_samples']:
                s_ind = np.random.choice(N, size=args['dbg_batch'], replace=False)
                s_X = X[s_ind, :]
                if args['var_in']:
                    s_Y_var_in = Y_var[s_ind, :]
                    s_Y_mean, s_Y_var = f_step(s_X, s_Y_var_in)
                else:
                    s_Y_mean, s_Y_var = f_step(s_X)
                err_plt = lambda: plt.errorbar(s_X[:, _ix], s_Y_mean[:, _iy],
                                               yerr=np.sqrt(s_Y_var[:, _iy]), fmt='none')
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
        _num = _grad_norm_cmp.shape[0]
        for _i in range(_num):
            plt.subplot(_num, 1, _i+1); plt.plot(conv_smoother(_grad_norm_cmp[_i]))
        plt.savefig(safe_path('norm_grad_cmp.png')); plt.close()
        # plot theta norm component-wise
        _theta_norm_cmp = np.array(it_theta_norm_comp).T
        plt.figure(); plt.suptitle('theta norm layer-wise')
        _num = _theta_norm_cmp.shape[0]
        for _i in range(_num):
            plt.subplot(_num, 1, _i+1); plt.plot(conv_smoother(_theta_norm_cmp[_i]))
        plt.savefig(safe_path('norm_theta_cmp.png')); plt.close()
        # plot samples for each epoch
        if args['dbg_plot_samples']:
            for _e, _distr in enumerate(ep_net_distr):
                _ttl = 'epoch_%d.png' % _e
                plt.scatter(X[:, _ix], Y[:, _iy], alpha=0.5, color='y', marker='*')
                plt.gca().set_autoscale_on(False)
                _distr[1](); plt.title(_ttl)
                plt.savefig(safe_path('_sample/' + _ttl)); plt.cla()
    return dbg_iter, dbg_done

if __name__ == "__main__":
    X, Y, Y_var = data_synthetic_a(1000)
    X, Y, Y_var = scale_data(X, Y, Y_var=Y_var)
    problem = create(DEFAULT_ARGS)
    step(X, Y, problem, DEFAULT_ARGS, Y_var=Y_var)
