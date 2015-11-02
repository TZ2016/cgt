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


def make_funcs(network, size_in, size_out):
    # step func
    X = cgt.matrix("X", fixed_shape=(None, size_in))
    Y_hat = network([X])[0]
    f_step = cgt.function([X], [Y_hat])
    # loss func
    Y = cgt.matrix("Y", fixed_shape=(None, size_out))
    size_batch = X.shape[0]
    loss = cgt.sum(cgt.norm(Y_hat - Y, axis=1)) / size_batch
    f_loss = cgt.function([X, Y], [Y_hat, loss])
    # grad func
    size_sample = 10  # number of samples for each example
    size_random = 10  # number of stochastic nodes
    size_costs = 1  # number of cost nodes
    # EM: for each example, calculate grad using samples and update

    C = cgt.matrix("C", fixed_shape=(size_sample, size_costs))
    H = cgt.matrix("H", fixed_shape=(size_sample, size_random))

    # TODO_TZ: f_grad requires graph conversion
    return f_step, f_loss, None


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_inputs", type=int)
    parser.add_argument("--num_units", type=int, nargs='+')
    parser.add_argument("--num_sto", type=int, nargs='+')
    args = parser.parse_args()

    # Hybrid layer with multiple stochastic nodes
    # And pure stochastic layer
    # --num_inputs 1 --num_units 3 2 --num_sto 2 2

    sfnn = hybrid_network(args.num_inputs, args.num_units, args.num_sto)
    f_step, f_loss, f_grad = make_funcs(sfnn, args.num_inputs, args.num_units[-1])
    for _ in range(10):
        # The number does not matter without training
        # The following features batch_size > 1
        print f_loss(np.array([[7], [8]]), np.array([[0, 0], [0, 0]]))
        # Verify that the output is stochastic

if __name__ == "__main__":
    main()
