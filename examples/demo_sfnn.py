# Learning Stochastic Feedforward Neural Networks

import cgt
from cgt import nn
import numpy as np


def hybrid_layer(X, size_in, size_out, size_random):
    assert size_out >= size_random
    out = cgt.sigmoid(nn.Affine(size_in, size_out)(X))
    if size_random == 0:
        return out, None
    else:
        out_s = cgt.bernoulli(out[:, :size_random])
        out = cgt.concatenate([out_s, out[:, size_random:]], axis=1)
        return out, out_s


def hybrid_network(size_in, num_units, num_stos):
    X = cgt.matrix(fixed_shape=(None, size_in))
    prev_num_units, prev_out = size_in, X
    all_sto_out, net_out_sto = [], None
    for (curr_num_units, curr_num_sto) in zip(num_units, num_stos):
        prev_out, sto_out = hybrid_layer(
            prev_out, prev_num_units, curr_num_units, curr_num_sto
        )
        if sto_out: all_sto_out.append(sto_out)
        prev_num_units = curr_num_units
    # TODO_TZ: missing output layer for the complete SFNN
    net_out = prev_out
    if all_sto_out: net_out_sto = cgt.concatenate(all_sto_out, axis=1)
    return X, net_out, net_out_sto


def sample(num_sample, in_values, out_values, net_in, net_out, net_out_rand):
    assert in_values.shape[0] == out_values.shape[0]
    rand_values = np.zeros((num_sample, )

    func = cgt.function([net_in], [net_out, net_out_rand])
    for i in xrange(num_sample):


def make_funcs(net_in, net_out, net_out_rand):
    size_batch = net_in.shape[0]
    # step func
    f_step = cgt.function([net_in], [net_out, net_out_rand])
    # loss func
    Y = cgt.matrix()
    loss = cgt.sum(cgt.norm(net_out - Y, axis=1)) / size_batch
    f_loss = cgt.function([net_in, Y], [net_out, net_out_rand, loss])
    # grad func
    size_sample = 10  # number of samples for each example
    size_random = 10  # number of stochastic nodes
    size_costs = 1  # number of cost nodes
    # EM: for each example, calculate grad using samples and update

    C = cgt.matrix("C", fixed_shape=(size_sample, size_costs))
    H = cgt.matrix("H", fixed_shape=(size_sample, size_random))

    # TODO_TZ: f_grad requires graph conversion
    return f_step, f_loss, None


def nice_print(X, Y, func, kind="loss"):
    assert X.shape[0] == Y.shape[0], "unequal batch size"
    if kind == "loss":
        net_out, net_out_rand, loss = func(X, Y)
        print "====="
        print net_out
        print loss
        print net_out_rand


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

    X, out, out_rand = hybrid_network(args.num_inputs, args.num_units, args.num_sto)
    f_step, f_loss, f_grad = make_funcs(X, out, out_rand)
    for _ in range(10):
        # The number does not matter without training
        # The following features batch_size > 1
        nice_print(np.array([[7], [8]]), np.array([[0, 0], [0, 0]]), f_loss)
        # Verify that the output is stochastic

if __name__ == "__main__":
    main()
