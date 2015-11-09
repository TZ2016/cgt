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
from demo_char_rnn import make_rmsprop_state, rmsprop_update, Table

def err_handler(type, flag):
    print "OH NOOOO!"
    print type, flag
    traceback.print_exc()
    # raise FloatingPointError('refer to err_handler for more details')

np.seterr(all='call')
np.seterrcall(err_handler)

DEFAULT_ARGS = Table(
    # network
    num_inputs=1,
    num_outputs=1,
    num_units=[4, 5, 5, 4],
    num_sto=[0, 2, 2, 0],
    no_bias=False,
    # training
    n_epochs=10,
    step_size=.01,
    decay_rate=.95,
)


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


def make_funcs(net_in, net_out):
    def f_grad (*x):
        out = f_surr(*x)
        return out['loss'], out['surr_loss'], out['surr_grad']
    Y = cgt.matrix("Y")
    size_batch = net_in.shape[0]
    # step func
    f_step = cgt.function([net_in], [net_out])
    # square loss
    # loss = cgt.sum((net_out - Y) ** 2) / size_batch
    # loglik of data
    size_out = Y.shape[1]
    out_sigma = cgt.exp(net_out[:, :size_out]) + 1.e-6  # positive sigma
    loss = -gaussian_diagonal.loglik(
        Y, net_out[:, :size_out], out_sigma
    ) / size_batch
    params = nn.get_parameters(loss)
    if DEFAULT_ARGS.no_bias:
        params = [p for p in params if not p.name.endswith(".b")]
    f_loss = cgt.function([net_in, Y], [net_out, loss])
    # grad func
    f_surr = get_surrogate_func([net_in, Y], [net_out], loss, params)
    return params, f_step, f_loss, f_grad, f_surr


def nice_print(X, Y, func):
    assert X.shape[0] == Y.shape[0], "unequal batch size"
    _out = func(X, Y)
    print "====="
    if isinstance(_out, (list, tuple)):
        for _o in _out:
            print _o
    elif isinstance(_out, dict):
        for k, v in _out.iteritems():
            print (k, v)
    else:
        print _out


def main():
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--step_size", type=float, default=.01)
    # parser.add_argument("--n_epochs", type=int, default=100)
    # parser.add_argument("--decay_rate", type=float, default=.95)
    # parser.add_argument("--num_inputs", type=int)
    # parser.add_argument("--num_units", type=int, nargs='+')
    # parser.add_argument("--num_sto", type=int, nargs='+')
    # args = parser.parse_args()
    args = DEFAULT_ARGS

    # Hybrid layer with multiple stochastic nodes
    # And pure stochastic layer
    # --num_inputs 1 --num_units 3 2 --num_sto 2 2

    net_in, net_out = hybrid_network(args.num_inputs, 2*args.num_outputs,
                                     args.num_units, args.num_sto)
    params, f_step, f_loss, f_grad, f_surr = make_funcs(net_in, net_out)
    param_col = ParamCollection(params)
    param_col.set_value_flat(
        np.random.uniform(-.1, .1, size=(param_col.get_total_size(),))
    )
    optim_state = make_rmsprop_state(theta=param_col.get_value_flat(),
                                     step_size=args.step_size,
                                     decay_rate=args.decay_rate)

    # X, Y = generate_examples(10, np.array([3.]), np.array([0.]), [.1])
    X, Y = data_synthetic_a(1000)

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

    all_loss = []
    all_surr_loss = []
    for i_epoch in range(args.n_epochs):
        for j in range(X.shape[0]):
            x, y = X[j:j+1], Y[j:j+1]
            info = f_surr(x, y)
            loss, loss_surr, grad = info['loss'], info['surr_loss'], info['surr_grad']
            # loss, loss_surr, grad = f_grad(x, y)
            all_loss.append(np.sum(loss))
            all_surr_loss.append(loss_surr)
            # update
            grad = param_col.flatten_values(grad)
            # rmsprop_update(grad, optim_state)
            optim_state.theta -= optim_state.step_size * grad
            print np.linalg.norm(grad)
            param_col.set_value_flat(optim_state.theta)
        print "Epoch %d" % i_epoch
        print "network parameters"
        _params_val =  param_col.get_values()
        # _ber_param = _params_val[0].T.dot(EXAMPLES_ARGS.x)
        # if not args.no_bias: _ber_param += _params_val[1].flatten()
        # _ber_param = sigmoid(_ber_param)
        # print ""
        # print "network params"
        # pprint.pprint(_params_val)
        # print "bernoulli param"
        # pprint.pprint( _ber_param)
        pprint.pprint(_params_val)
    all_loss, all_surr_loss = np.array(all_loss), np.array(all_surr_loss)
    plt.plot(np.convolve(all_loss, [1. / X.shape[0]] * X.shape[0], 'same'))
    plt.plot(np.convolve(all_surr_loss, [1. / X.shape[0]] * X.shape[0], 'same'))

if __name__ == "__main__":
    main()
