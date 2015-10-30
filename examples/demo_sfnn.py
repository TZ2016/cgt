# Learning Stochastic Feedforward Neural Networks

import cgt
from cgt import nn
import numpy as np


def hybrid_layer(size_in, size_out, size_random):
    assert size_out >= size_random
    X = cgt.matrix(fixed_shape=(None, size_in))
    out = cgt.sigmoid(nn.Affine(size_in, size_out)(X))
    if size_random > 0:
        out = cgt.concatenate([
            cgt.bernoulli(out[:, :size_random]),
            out[:, size_random:],
        ], axis=1)
    return nn.Module([X], [out])


def hybrid_network(size_in, num_units, num_sto):
    X = cgt.matrix(fixed_shape=(None, size_in))
    prev_num_units, prev_out = size_in, [X]
    for (num_units, num_sto) in zip(num_units, num_sto):
        layer = hybrid_layer(prev_num_units, num_units, num_sto)
        prev_num_units, prev_out = num_units, layer(prev_out)
    # TODO_TZ: missing output layer
    return nn.Module([X], prev_out)


def make_funcs(network, size_in):
    X = cgt.matrix("X", fixed_shape=(1, size_in))
    f_step = cgt.function([X], network([X]))
    # TODO_TZ: f_loss is easier
    # TODO_TZ: f_grad requires graph conversion
    return f_step, None, None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_inputs", type=int)
    parser.add_argument("--num_units", type=int, nargs='+')
    parser.add_argument("--num_sto", type=int, nargs='+')
    args = parser.parse_args()

    # TODO_TZ: not support multiple stochastic nodes in one layer

    sfnn = hybrid_network(args.num_inputs, args.num_units, args.num_sto)
    f_step, f_loss, f_grad = make_funcs(sfnn, args.num_inputs)
    for _ in range(10):
        print f_step(np.array([[7]]))


if __name__ == "__main__":
    main()
