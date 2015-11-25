# Learning Stochastic Feedforward Neural Networks

import os
import cgt
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

def step(X, Y, workspace, config, Y_var=None, dbg_iter=None, dbg_done=None):
    if config['debug'] and (dbg_iter is None or dbg_done is None):
        dbg_iter, dbg_done = example_debug(config, X, Y, Y_var=Y_var)
    if config['var_in']: assert Y_var is not None
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
        workspace['update'](param_col.flatten_values(grad), optim_state)
        param_col.set_value_flat(optim_state['theta'])
        num_iters += 1
        if num_iters == Y.shape[0]:
            num_epochs += 1
            num_iters = 0
        if dbg_iter:
            dbg_iter(num_epochs, num_iters, info, workspace)
    if dbg_done: dbg_done(workspace)
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
        if optim_state['type'] == 'adam':
            f_update = adam_update
        elif optim_state['type'] == 'rmsprop':
            f_update = rmsprop_update
        else:
            raise ValueError
    else:
        theta = param_col.get_value_flat()
        method = args['opt_method'].lower()
        if method == 'rmsprop':
            optim_state = rmsprop_create(theta, step_size=args['step_size'])
            f_update = rmsprop_update
        elif method == 'adam':
            optim_state = adam_create(theta, step_size=args['step_size'])
            f_update = adam_update
        else:
            raise ValueError('unknown optimization method: %s' % method)
        init_method = args['init_conf']
        if init_method == 'XavierNormal':
            init_params = nn.XavierNormal(**args['init_conf_params'])
        elif init_method == 'gaussian':
            init_params = nn.IIDGaussian(**args['init_conf_params'])
        else:
            raise ValueError('unknown init distribution')
        optim_state['theta'] = nn.init_array(
            init_params, (param_col.get_total_size(), 1)).flatten()
    param_col.set_value_flat(optim_state['theta'])
    print "Initialization"
    pprint.pprint(param_col.get_values())
    workspace = {
        'optim_state': optim_state,
        'param_col': param_col,
        'f_surr': f_surr,
        'f_step': f_step,
        'f_loss': f_loss,
        'f_grad': f_grad,
        'update': f_update,
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
    def h_ax(ax, title=None, x=None, y=None):
        if not isinstance(ax, (tuple, list, np.ndarray)): ax = [ax]
        for a in ax:
            if title: a.set_title(title)
            if x:
                if x == 'hide':
                    plt.setp(a.get_xticklabels(), visible=False)
                elif x == 'mm':  # min/max
                    plt.setp(a.get_xticklabels()[1:-1], visible=False)
                    plt.setp(a.get_xticklabels()[0], visible=True)
                    plt.setp(a.get_xticklabels()[-1], visible=True)
                else:
                    raise KeyError
            if y:
                if y == 'hide':
                    plt.setp(a.get_yticklabels(), visible=False)
                elif y == 'mm':  # min/max
                    plt.setp(a.get_yticklabels()[1:-1], visible=False)
                    plt.setp(a.get_yticklabels()[0], visible=True)
                    plt.setp(a.get_yticklabels()[-1], visible=True)
                else:
                    raise KeyError
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
    def dbg_iter(num_epochs, num_iters, info, workspace):
        optim_state = workspace['optim_state']
        param_col = workspace['param_col']
        f_step = workspace['f_step']
        it_loss_surr.append(info['objective'])
        it_grad_norm.append(np.linalg.norm(optim_state['scratch']))
        it_grad_norm_comp.append([np.linalg.norm(g) / np.size(g)
                                  for g in info['grad']])
        it_theta_norm.append(np.linalg.norm(optim_state['theta']))
        it_theta_norm_comp.append([np.linalg.norm(t) / np.size(t)
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
    def dbg_done(workspace):
        kw_ticks = {
            'xticks': np.arange(args['n_epochs']) * N,
            'xticklabels': np.arange(args['n_epochs']).astype(str)
        }
        param_col = workspace['param_col']
        optim_state = workspace['optim_state']
        # save params
        pickle.dump(args, open(safe_path('args.pkl'), 'w'))
        pickle.dump(param_col.get_values(), open(safe_path('params.pkl'), 'w'))
        pickle.dump(optim_state, open(safe_path('__snapshot.pkl'), 'w'))
        pickle.dump(np.array(it_theta_comp), open(safe_path('params_history.pkl'), 'w'))
        # plot overview
        f, axs = plt.subplots(3, 1, sharex=True, subplot_kw=kw_ticks)
        f.suptitle('overview')
        axs[0].plot(conv_smoother(it_loss_surr))
        h_ax(axs[0], title='loss', x='hide')
        axs[1].plot(conv_smoother(it_grad_norm)); axs[1].set_title('grad')
        h_ax(axs[1], title='grad', x='hide')
        axs[2].plot(conv_smoother(it_theta_norm)); axs[2].set_title('theta')
        h_ax(axs[2], title='theta')
        f.savefig(safe_path('overview.png')); plt.close(f)
        # plot grad norm component-wise
        _grad_norm_cmp = np.array(it_grad_norm_comp).T
        f, axs = plt.subplots(_grad_norm_cmp.shape[0], 1, sharex=True, subplot_kw=kw_ticks)
        f.suptitle('grad norm layer-wise')
        for _i, _ax in enumerate(axs):
            _ax.plot(conv_smoother(_grad_norm_cmp[_i]))
        h_ax(axs[:-1], x='hide'); h_ax(axs, y='mm')
        f.tight_layout(); f.savefig(safe_path('norm_grad_cmp.png')); plt.close(f)
        # plot theta norm component-wise
        _theta_norm_cmp = np.array(it_theta_norm_comp).T
        f, axs = plt.subplots(_theta_norm_cmp.shape[0], 1, sharex=True, subplot_kw=kw_ticks)
        f.suptitle('theta norm layer-wise')
        for _i, _ax in enumerate(axs):
            _ax.plot(conv_smoother(_theta_norm_cmp[_i]))
        h_ax(axs[:-1], x='hide'); h_ax(axs, y='mm')
        f.tight_layout(); f.savefig(safe_path('norm_theta_cmp.png')); plt.close(f)
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
    import yaml
    import time

    DUMP_ROOT = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_tmp')
    PARAMS_PATH = os.path.join(DUMP_ROOT, '../sfnn_params.yaml')
    DEFAULT_ARGS = yaml.load(open(PARAMS_PATH, 'r'))
    DEFAULT_ARGS['dump_path'] = os.path.join(DUMP_ROOT,'_%d/' % int(time.time()))
    print "Default args:"
    pprint.pprint(DEFAULT_ARGS)

    X, Y, Y_var = data_synthetic_a(1000)
    X, Y, Y_var = scale_data(X, Y, Y_var=Y_var)
    problem = create(DEFAULT_ARGS)
    step(X, Y, problem, DEFAULT_ARGS, Y_var=Y_var)
