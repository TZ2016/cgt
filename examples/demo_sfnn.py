# Learning Stochastic Feedforward Neural Networks

import cgt
import pprint
from cgt.core import get_surrogate_func
from cgt import nn
import numpy as np
from scipy.special import expit as sigmoid
from param_collection import ParamCollection
from demo_char_rnn import make_rmsprop_state, rmsprop_update, Table

EXAMPLES_ARGS = Table(
    # example generation
    num_examples=10,
    x=np.array([3.]),
    y=np.array([0.]),
    truth_ratio=[.1],
)

DEFAULT_ARGS = Table(
    # network
    num_inputs=EXAMPLES_ARGS.x.size,
    num_outputs=EXAMPLES_ARGS.y.size,
    num_units=[1],
    num_sto=[1],
    no_bias=True,
    # training
    n_epochs=100,
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
    # net_out = nn.Affine(prev_num_units, size_out,
    #                     name="InnerProd(%d->%d)" % (prev_num_units, size_out)
    #                     )(prev_out)
    assert prev_num_units == size_out
    net_out = prev_out
    return X, net_out


def make_funcs(net_in, net_out):
    def f_grad (*x):
        out = f_surr(*x)
        return out['surr_grad']
    size_batch = net_in.shape[0]
    # step func
    f_step = cgt.function([net_in], [net_out])
    # loss func
    Y = cgt.matrix("Y")
    # loss = cgt.sum(cgt.norm(net_out - Y, axis=1)) / size_batch
    loss = cgt.sum((net_out - Y) ** 2) / size_batch
    params = nn.get_parameters(loss)
    if DEFAULT_ARGS.no_bias:
        params = [p for p in params if not p.name.endswith(".b")]
    # loss = cgt.sum(net_out - Y)  # this is for debugging use
    f_loss = cgt.function([net_in, Y], [net_out, loss])
    # grad func
    f_surr = get_surrogate_func([net_in, Y], [net_out], loss, params)
    return params, f_step, f_loss, f_grad


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

    X, out = hybrid_network(args.num_inputs, args.num_outputs, args.num_units, args.num_sto)
    params, f_step, f_loss, f_grad = make_funcs(X, out)
    param_col = ParamCollection(params)
    param_col.set_value_flat(
        np.random.uniform(2., 2., size=(param_col.get_total_size(),))
    )
    optim_state = make_rmsprop_state(theta=param_col.get_value_flat(),
                                     step_size=args.step_size,
                                     decay_rate=args.decay_rate)

    training_x, training_y = generate_examples(EXAMPLES_ARGS.num_examples,
                                               EXAMPLES_ARGS.x, EXAMPLES_ARGS.y,
                                               EXAMPLES_ARGS.truth_ratio)

    for i_epoch in range(args.n_epochs):
        for j in range(training_x.shape[0]):
            x, y = training_x[j:j+1], training_y[j:j+1]
            grad = f_grad(x, y)
            # print grad
            grad = param_col.flatten_values(grad)
            rmsprop_update(grad, optim_state)
            param_col.set_value_flat(optim_state.theta)
            _params_val =  param_col.get_values()
            _ber_param = _params_val[0].T.dot(EXAMPLES_ARGS.x)
            if not args.no_bias: _ber_param += _params_val[1]
            _ber_param = sigmoid(_ber_param)
            print ""
            print "network params"
            pprint.pprint(_params_val)
            print "bernoulli param"
            pprint.pprint( _ber_param)
            # print f_step(x)
            # print scipy.special.expit(param_col.get_values()[0] * DEFAULT_ARGS.x)
        # nice_print(np.array([[7], [8]]), np.array([[0, 0], [0, 0]]), f_grad)

if __name__ == "__main__":
    main()
