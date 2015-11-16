# Learning Stochastic Feedforward Neural Networks

import os
import cgt
import time
import warnings
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
from demo_char_rnn import Table, make_rmsprop_state


def err_handler(type, flag):
    print type, flag
    # raise FloatingPointError('refer to err_handler for more details')
np.seterr(divide='call', over='call', invalid='call')
np.seterrcall(err_handler)
np.set_printoptions(precision=4, suppress=True)
print cgt.get_config(True)


def generate_examples(N, x, y, p_y):
    X = x * np.ones((N, x.size))
    Y = y * np.ones((N, y.size))
    for i, p in enumerate(p_y):
        if p is not None:
            Y[:, i] = 0.
            Y[:, i][:int(N*p)] = 1.
    np.random.shuffle(Y)
    return X, Y


def scale_data(Xs, scalars=None):
    if not scalars:
        scalars = [StandardScaler() for _ in range(len(Xs))]
    assert len(scalars) == len(Xs)
    Xs = [scalar.fit_transform(X) for X, scalar in zip(Xs, scalars)]
    return Xs


def data_synthetic_a(N):
    # x = y + 0.3 sin(2 * pi * y) + e, e ~ Unif(-0.1, 0.1)
    Y = np.random.uniform(0., 1., N)
    X = Y + 0.3 * np.sin(2. * Y * np.pi) + np.random.uniform(-.1, .1, N)
    Y, X = Y.reshape((N, 1)), X.reshape((N, 1))
    return X, Y


def data_simple_sigmoid(N):
    X = np.random.uniform(-10., 10., N)
    Y = sigmoid(X)
    Y += np.random.normal(0., .1, N)
    Y += np.random.binomial(1, .3, N)
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
    def f_grad (*x):
        out = f_surr(*x)
        return out['loss'], out['surr_loss'], out['grad']
    Y = cgt.matrix("Y")
    params = nn.get_parameters(net_out)
    if 'no_bias' in config and config['no_bias']:
        print "Excluding bias"
        params = [p for p in params if not p.name.endswith(".b")]
    size_out, size_batch = Y.shape[1], net_in.shape[0]
    f_step = cgt.function([net_in], [net_out])
    # loss_raw of shape (size_batch, 1); loss should be a scalar
    # sum-of-squares loss
    sigma = 0.1
    loss_raw = -.5 * cgt.sum((net_out - Y) ** 2, axis=1, keepdims=True) / sigma
    # negative log-likelihood
    # out_sigma = cgt.exp(net_out[:, size_out:]) + 1.e-6  # positive sigma
    # loss_raw = -gaussian_diagonal.logprob(
    #     Y, net_out,
        # out_sigma
        # cgt.fill(.01, [size_batch, size_out])
    # )
    if 'param_penal_wt' in config:
        print "Applying penalty on parameter norm"
        assert config['param_penal_wt'] > 0
        params_flat = cgt.concatenate([p.flatten() for p in params])
        loss_param = cgt.fill(cgt.sum(params_flat ** 2), [size_batch, 1])
        loss_param *= config['param_penal_wt']
        loss_raw += loss_param
    loss = cgt.sum(loss_raw) / size_batch
    # end of loss definition
    f_loss = cgt.function([net_in, Y], [net_out, loss])
    f_surr = get_surrogate_func([net_in, Y],
                                [net_out] + dbg_out,
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
    # TIANHAO_TZ double check "+=" is the only change


def train(args, X, Y, dbg_iter=None, dbg_epoch=None, dbg_done=None):
    dbg_out = []
    net_in, net_out = hybrid_network(args.num_inputs, args.num_outputs,
                                     args.num_units, args.num_sto,
                                     dbg_out=dbg_out)
    params, f_step, f_loss, f_grad, f_surr = \
        make_funcs(net_in, net_out, args, dbg_out=dbg_out)
    param_col = ParamCollection(params)
    init_params = nn.init_array(args.init_conf, (param_col.get_total_size(), 1))
    param_col.set_value_flat(init_params.flatten())
    if 'snapshot' in args:
        print "Loading params from previous snapshot"
        snapshot = pickle.load(open(args['snapshot'], 'r'))
        param_col.set_values(snapshot)
    optim_state = make_rmsprop_state(theta=param_col.get_value_flat(),
                                     step_size=args.step_size,
                                     decay_rate=args.decay_rate)
    for i_epoch in range(args.n_epochs):
        for i_iter in range(X.shape[0]):
            ind = np.random.choice(X.shape[0], args['size_batch'])
            x, y = X[ind], Y[ind]  # not sure this works for multi-dim
            info = f_surr(x, y, num_samples=args['size_sample'])
            grad = info['grad']
            # update
            rmsprop_update(param_col.flatten_values(grad), optim_state)
            param_col.set_value_flat(optim_state.theta)
            if dbg_iter: dbg_iter(i_epoch, i_iter, param_col, optim_state, info)
        if dbg_epoch: dbg_epoch(i_epoch, param_col, f_surr)
    if dbg_done: dbg_done(param_col, optim_state, f_surr)
    return optim_state


def example_debug(args, X, Y, out_path='.'):
    def safe_path(rel_path):
        abs_path = os.path.join(out_path, rel_path)
        d = os.path.dirname(abs_path)
        if not os.path.exists(d):
            warnings.warn("Making new directory: %s" % d)
            os.makedirs(d)
        return abs_path
    N, _ = X.shape
    plt_markers = ['x', 'o', 'v', 's', '+', '*']
    plt_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    plt_kwargs = [{'marker': m, 'color': c, 's': 10}
                  for m in plt_markers for c in plt_colors]
    size_sample, max_plots = args['dbg_samples'], 5
    conv_smoother = lambda x: np.convolve(x, [1. / N] * N, mode='valid')
    # cache
    ep_samples = []
    it_loss, it_loss_surr = [], []
    it_grad_norm, it_theta_norm = [], []
    it_grad_norm_comp, it_theta_comp = [], []
    def dbg_iter(i_epoch, i_iter, param_col, optim_state, info):
        it_loss.append(np.sum(info['objective_unweighted']))
        it_loss_surr.append(info['objective'])
        it_grad_norm.append(np.linalg.norm(optim_state.scratch))
        it_theta_norm.append(np.linalg.norm(optim_state.theta))
        it_theta_comp.append(np.copy(optim_state.theta))
        it_grad_norm_comp.append([np.linalg.norm(g) for g in info['grad']])
    def dbg_epoch(i_epoch, param_col, f_surr):
        print "Epoch %d" % i_epoch
        print "network parameters"
        _params_val = param_col.get_values()
        pprint.pprint(_params_val)
        # sample the network to track progress
        s_X = np.random.choice(X.flatten(), size=(size_sample, 1), replace=False)
        info = f_surr(s_X, np.zeros_like(s_X), sample_only=True, num_samples=1)
        s_Y = info['outputs'][0]
        ep_samples.append((i_epoch, s_X.flatten(), s_Y.flatten()))
    def dbg_done(param_col, optim_state, f_surr):
        plt.close('all')
        # plot samples together
        _ep_samples = np.array(ep_samples, dtype='object')
        _max_plots = min(len(ep_samples), max_plots)
        _split = [l[0] for l in np.array_split(_ep_samples, _max_plots)]
        _plots = [plt.scatter(*s[1:], **kw) for s, kw in zip(_split, plt_kwargs)]
        plt.legend(_plots, [l[0] for l in _split], scatterpoints=1, fontsize=6)
        plt.title('samples from network'); plt.savefig(safe_path('net_samples.png'))
        # plot samples for each epoch
        _axis = plt.axis()  # this range should be good
        for _e, _sample in enumerate(ep_samples):
            _ttl = 'epoch_%d.png' % _e
            plt.cla(); plt.scatter(*_sample[1:]); plt.axis(_axis); plt.title(_ttl)
            plt.scatter(X, Y, alpha=0.5, color='y', marker='*')
            plt.savefig(safe_path('_sample/' + _ttl))
        # plot norms
        plt.close(); plt.figure(); plt.suptitle('norm')
        plt.subplot(211); plt.plot(conv_smoother(it_grad_norm)); plt.title('grad')
        plt.subplot(212); plt.plot(conv_smoother(it_theta_norm)); plt.title('theta')
        plt.savefig(safe_path('norm.png'))
        # plot grad norm component-wise
        _norm_cmp = np.array(it_grad_norm_comp).T
        plt.close(); plt.figure(); plt.suptitle('grad norm layer-wise')
        _num = len(it_grad_norm_comp[0])
        for _i in range(_num):
            plt.subplot(_num, 1, _i+1); plt.plot(conv_smoother(_norm_cmp[_i]))
        plt.savefig(safe_path('norm_grad_cmp.png'))
        # plot theta component-wise
        _theta_cmp = np.array(it_theta_comp).T
        plt.close(); plt.figure(); plt.suptitle('theta components')
        _num = len(it_theta_comp[0])
        for _i in range(_num):
            plt.subplot(_num, 1, _i+1); plt.plot(conv_smoother(_theta_cmp[_i]))
        plt.savefig(safe_path('theta_cmp.png'))
        # plot loss
        plt.close(); plt.figure(); plt.suptitle('loss')
        plt.subplot(211); plt.plot(conv_smoother(it_loss)); plt.title('orig')
        plt.subplot(212); plt.plot(conv_smoother(it_loss_surr)); plt.title('surr')
        plt.savefig(safe_path('loss.png'))
        # save params
        theta = param_col.get_values()
        pickle.dump(theta, open(safe_path('params.pkl'), 'w'))
        pickle.dump(args, open(safe_path('args.pkl'), 'w'))
    return {'dbg_iter': dbg_iter, 'dbg_epoch': dbg_epoch, 'dbg_done': dbg_done}

if __name__ == "__main__":
    DUMP_PATH = os.path.join(os.environ['HOME'],
                             'workspace/cgt/tmp/_%d' % int(time.time()))
    #################3#####
    #  for synthetic data #
    #######################
    example_args = Table(
        # network architecture
        num_inputs=1,
        num_outputs=1,
        num_units=[2],
        num_sto=[1],
        no_bias=False,
        # training parameters
        n_epochs=50,
        step_size=.01,
        decay_rate=.95,
        size_sample=20,  # #times to sample the network per data pair
        size_batch=1,  # #data pairs for each gradient estimate
        init_conf=nn.XavierNormal(scale=1.),
        # param_penal_wt=1.e-4,
        # snapshot=os.path.join(DUMP_PATH, 'params.pkl')
        # debugging
        dbg_samples=100,
    )
    X, Y = data_synthetic_a(500)
    # X, Y = scale_data((X, Y))
    state = train(example_args, X, Y,
                  **example_debug(example_args, X, Y, DUMP_PATH))

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
