# Learning Stochastic Feedforward Neural Networks

import cgt
import pprint
import matplotlib.pyplot as plt
from cgt.core import get_surrogate_func
from cgt import nn
import numpy as np
import traceback
from scipy.special import expit as sigmoid
from param_collection import ParamCollection
from cgt.distributions import gaussian_diagonal
from demo_char_rnn import Table


def err_handler(type, flag):
    print type, flag
    traceback.print_exc()
    # raise FloatingPointError('refer to err_handler for more details')
np.seterr(all='call')
np.seterrcall(err_handler)


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


def data_simple(N):
    X = np.random.uniform(0., 1., N)
    Y = X + np.random.uniform(-.1, .1, N)
    # Y += np.random.binomial(1, 0.5, N)
    Y, X = Y.reshape((N, 1)), X.reshape((N, 1))
    return X, Y


def hybrid_layer(X, size_in, size_out, size_random):
    assert size_out >= size_random >= 0
    out = cgt.sigmoid(nn.Affine(
        size_in, size_out, name="InnerProd(%d->%d)" % (size_in, size_out)
    )(X))
    if size_random == 0:
        return out
    if size_random == size_out:
        out_s = cgt.bernoulli(out)
        return out_s
    out_s = cgt.bernoulli(out[:, :size_random])
    out = cgt.concatenate([out_s, out[:, size_random:]], axis=1)
    return out


def hybrid_network(size_in, size_out, num_units, num_stos):
    X = cgt.matrix("X", fixed_shape=(None, size_in))
    prev_num_units, prev_out = size_in, X
    for (curr_num_units, curr_num_sto) in zip(num_units, num_stos):
        prev_out = hybrid_layer(
            prev_out, prev_num_units, curr_num_units, curr_num_sto
        )
        prev_num_units = curr_num_units
    # TODO_TZ bigger problem! param cannot deterministically influence cost
    #         otherwise the surrogate cost is not complete log likelihood
    net_out = nn.Affine(prev_num_units, size_out,
                        name="InnerProd(%d->%d)" % (prev_num_units, size_out)
                        )(prev_out)
    # assert prev_num_units == size_out
    # net_out = prev_out
    return X, net_out


def make_funcs(net_in, net_out, **kwargs):
    def f_grad (*x):
        out = f_surr(*x)
        return out['loss'], out['surr_loss'], out['surr_grad']
    Y = cgt.matrix("Y")
    size_out, size_batch = Y.shape[1], net_in.shape[0]
    f_step = cgt.function([net_in], [net_out])
    # loss_raw of shape (size_batch, 1); loss should be a scalar
    # sum-of-squares loss
    loss_raw = cgt.sum((net_out - Y) ** 2, axis=1, keepdims=True)
    # negative log-likelihood
    # out_sigma = cgt.exp(net_out[:, size_out:]) + 1.e-6  # positive sigma
    # loss_raw = -gaussian_diagonal.logprob(
    #     Y, net_out[:, :size_out],
    #     out_sigma  # cgt.fill(.01, [size_batch, 1])
    # )
    loss = cgt.sum(loss_raw) / size_batch
    # end of loss definition
    params = nn.get_parameters(loss)
    if kwargs.has_key('no_bias'):
        params = [p for p in params if not p.name.endswith(".b")]
    f_loss = cgt.function([net_in, Y], [net_out, loss])
    f_surr = get_surrogate_func([net_in, Y], [net_out], loss_raw, params, 40)
    return params, f_step, f_loss, f_grad, f_surr


def train(args, X, Y, dbg_iter=None, dbg_epoch=None, dbg_done=None):
    net_in, net_out = hybrid_network(args.num_inputs, args.num_outputs,
                                     args.num_units, args.num_sto)
    params, f_step, f_loss, f_grad, f_surr = make_funcs(net_in, net_out)
    param_col = ParamCollection(params)
    param_col.set_value_flat(
        np.random.uniform(-.1, .1, size=(param_col.get_total_size(),))
    )
    optim_state = Table(theta=param_col.get_value_flat(),
                        grad=param_col.get_value_flat(),
                        step_size=args.step_size
                        )
    all_loss, all_surr_loss = [], []
    for i_epoch in range(args.n_epochs):
        for i_iter in range(X.shape[0]):
            x, y = X[i_iter:i_iter+1], Y[i_iter:i_iter+1]
            info = f_surr(x, y)
            loss, loss_surr, grad = info['loss'], info['surr_loss'], info['surr_grad']
            # loss, loss_surr, grad = f_grad(x, y)
            all_loss.append(np.sum(loss))
            all_surr_loss.append(loss_surr)
            # update
            optim_state.grad = param_col.flatten_values(grad)
            optim_state.theta -= optim_state.step_size * optim_state.grad
            param_col.set_value_flat(optim_state.theta)
            if dbg_iter: dbg_iter(i_epoch, i_iter, optim_state, info)
        if dbg_epoch: dbg_epoch(i_epoch, param_col, f_surr)
    if dbg_done: dbg_done()
    all_loss, all_surr_loss = np.array(all_loss), np.array(all_surr_loss)
    plt.plot(np.convolve(all_loss, [1. / X.shape[0]] * X.shape[0], 'same'))
    plt.plot(np.convolve(all_surr_loss, [1. / X.shape[0]] * X.shape[0], 'same'))


def example_debug():
    plt_markers = ['x', 'o', 'v', 's', '+', '*']
    plt_colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    plt_kwargs = [{'marker': m, 'color': c, 's': 10}
                  for m in plt_markers for c in plt_colors]
    size_sample, max_plots = 50, 10
    ep_samples = []
    def dbg_iter(i_epoch, i_iter, optim_state, info):
        print np.linalg.norm(optim_state.grad)
    def dbg_epoch(i_epoch, param_col, f_surr):
        print "Epoch %d" % i_epoch
        print "network parameters"
        _params_val = param_col.get_values()
        # _ber_param = _params_val[0].T.dot(EXAMPLES_ARGS.x)
        # if not args.no_bias: _ber_param += _params_val[1].flatten()
        # _ber_param = sigmoid(_ber_param)
        # print ""
        # print "network params"
        # pprint.pprint(_params_val)
        # print "bernoulli param"
        # pprint.pprint( _ber_param)
        pprint.pprint(_params_val)
        # sample the network to track progress
        s_X = np.random.uniform(0., 1., (size_sample, 1))
        info = f_surr(s_X, np.zeros((size_sample, 1)), no_sample=True)
        s_Y = info['net_out'][0]
        ep_samples.append((i_epoch, (s_X.flatten(), s_Y.flatten())))
        # plot = plt.scatter(s_X.flatten(), s_Y.flatten(), **plt_kwargs[i_epoch])
        # s_Y_mu, s_Y_var = s_Y[:, 0], np.exp(s_Y[:, 1]) + 1.e-6
        # plt.scatter(s_X.flatten(), s_Y_mu.flatten())
        # ep_samples.append(plot)
    def dbg_done():
        assert len(ep_samples) > max_plots
        ep_samples_split = [l[0] for l in np.array_split(np.array(ep_samples), max_plots)]
        _plots = []
        for i, s in enumerate(ep_samples_split):
            _plots.append(plt.scatter(*s[1], **plt_kwargs[i]))
        plt.legend(_plots, [l[0] for l in ep_samples_split], scatterpoints=1, fontsize=6)
        plt.savefig('tmp.png')
    return {'dbg_iter': dbg_iter, 'dbg_epoch': dbg_epoch, 'dbg_done': dbg_done}

if __name__ == "__main__":
    #################3#####
    #  for synthetic data #
    #######################
    args_synthetic = Table(
        num_inputs=1,
        num_outputs=1,
        num_units=[1, ],
        num_sto=[0, ],
        no_bias=False,
        n_epochs=40,
        step_size=.1,
    )
    X_syn, Y_syn = data_synthetic_a(100)
    train(args_synthetic, X_syn, Y_syn, **example_debug())

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
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--step_size", type=float, default=.01)
    # parser.add_argument("--n_epochs", type=int, default=100)
    # parser.add_argument("--decay_rate", type=float, default=.95)
    # parser.add_argument("--num_inputs", type=int)
    # parser.add_argument("--num_units", type=int, nargs='+')
    # parser.add_argument("--num_sto", type=int, nargs='+')
    # args = parser.parse_args()


