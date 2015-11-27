from __future__ import division
import cgt
import time
from cgt import nn, utils
import numpy as np
from cgt.utility.param_collection import ParamCollection
from opt import rmsprop_create, rmsprop_update


# def cgt_gru(size_input, size_mem, n_layers, size_output, size_batch):
#     inputs = [cgt.matrix() for i_layer in xrange(n_layers+1)]
#     outputs = []
#     for i_layer in xrange(n_layers):
#         prev_h = inputs[i_layer+1] # note that inputs[0] is the external input, so we add 1
#         x = inputs[0] if i_layer==0 else outputs[i_layer-1]
#         size_x = size_input if i_layer==0 else size_mem
#         update_gate = cgt.sigmoid(
#             nn.Affine(size_x, size_mem,name="i2u")(x)
#             + nn.Affine(size_mem, size_mem, name="h2u")(prev_h))
#         reset_gate = cgt.sigmoid(
#             nn.Affine(size_x, size_mem,name="i2r")(x)
#             + nn.Affine(size_mem, size_mem, name="h2r")(prev_h))
#         gated_hidden = reset_gate * prev_h
#         p2 = nn.Affine(size_mem, size_mem)(gated_hidden)
#         p1 = nn.Affine(size_x, size_mem)(x)
#         hidden_target = cgt.tanh(p1+p2)
#         next_h = (1.0-update_gate)*prev_h + update_gate*hidden_target
#         outputs.append(next_h)
#     category_activations = nn.Affine(size_mem, size_output,name="pred")(outputs[-1])
#     logprobs = nn.logsoftmax(category_activations)
#     outputs.append(logprobs)
#
#     return nn.Module(inputs, outputs)


def lstm_block(h_prev, c_prev, x_curr, size_x, size_c):
    """
    Construct a LSTM cell block of specified number of cells

    :param h_prev: self activations at previous time step
    :param c_prev: self memory state at previous time step
    :param x_curr: inputs from previous layer at current time step
    :param size_x: size of inputs
    :param size_c: size of both c and h
    :return: c and h at current time step
    :rtype:
    """
    input_sums = nn.Affine(size_x, 4 * size_c)(x_curr) + \
                 nn.Affine(size_x, 4 * size_c)(h_prev)
    c_new = cgt.tanh(input_sums[:, 3*size_c:])
    sigmoid_chunk = cgt.sigmoid(input_sums[:, :3*size_c])
    in_gate = sigmoid_chunk[:, :size_c]
    forget_gate = sigmoid_chunk[:, size_c:2*size_c]
    out_gate = sigmoid_chunk[:, 2*size_c:3*size_c]
    c_curr = forget_gate * c_prev + in_gate * c_new
    h_curr = out_gate * cgt.tanh(c_curr)
    return c_curr, h_curr


def lstm_net(size_input, size_mem, n_layers, size_output, size_batch):
    """
    Construct a recurrent neural network with multiple layers of LSTM units,
    with each layer a block of cells sharing a common set of gate units.
    Return list of inputs and a list of outputs for the net at one time step.
    Inputs =  [ net_in, hidden layers ]
    Outputs = [ hidden layers, net_out ]

    :param size_input: input dimension
    :param size_mem: number of memory cells per layer
    :param n_layers: number of LSTM layers
    :param size_output: output dimension
    :param size_batch: batch size
    :return:
    :rtype: (int, list, list)
    """
    num_hidden = 2 * n_layers
    inputs = [cgt.matrix(fixed_shape=(size_batch, size_input))]
    for _ in xrange(2 * n_layers):
        inputs.append(cgt.matrix(fixed_shape=(size_batch, size_mem)))
    outputs = []
    for i_layer in xrange(n_layers):
        prev_h = inputs[i_layer * 2]
        prev_c = inputs[i_layer * 2 + 1]
        if i_layer == 0:
            x = inputs[0]
            size_x = size_input
        else:
            x = outputs[(i_layer-1)*2]
            size_x = size_mem
        next_c, next_h = lstm_block(prev_h, prev_c, x, size_x, size_mem)
        outputs.append(next_c)
        outputs.append(next_h)
    net_out = nn.Affine(size_mem, size_output)(outputs[-1])
    outputs.append(net_out)
    return num_hidden, inputs, outputs


class CGTNet(object):
    def __init__(self, net_cnst, loss_func,
                 n_layers, n_unroll,
                 size_input, size_output, size_mem, size_batch):
        self.net_cnst = net_cnst
        self.loss_func = loss_func
        self.n_layers = n_layers
        self.n_unroll = n_unroll
        self.size_input = size_input
        self.size_output = size_output
        self.size_mem = size_mem
        self.size_batch = size_batch

        x_tnk = cgt.tensor3(name="X")
        targ_tnk = cgt.tensor3(name="T")
        self.num_hidden_layers, net_ins_sym, net_outs_sym = \
            net_cnst(size_input, size_mem, n_layers, size_output, size_batch)
        self.net = nn.Module(net_ins_sym, net_outs_sym)

        loss = 0
        init_hiddens = [cgt.matrix() for _ in xrange(self.num_hidden_layers)]
        cur_hiddens = init_hiddens
        for t in xrange(n_unroll):
            net_outs = self.net([x_tnk[t]] + cur_hiddens)
            cur_hiddens, net_out = net_outs[:-1], net_outs[-1]
            loss = loss + loss_func(net_out, targ_tnk[t])
        assert len(init_hiddens) == len(cur_hiddens)
        loss = loss / (n_unroll * size_batch)

        params = self.net.get_parameters()
        self.params_col = ParamCollection(params)
        self.params_col.set_value_flat(
            np.random.uniform(-.1, .1, size=(self.params_col.get_total_size(),))
        )

        gradloss = cgt.grad(loss, params)
        flatgrad = cgt.concatenate([x.flatten() for x in gradloss])
        with utils.Message("compiling loss+grad"):
            self.f_loss_and_grad = cgt.function([x_tnk, targ_tnk] + init_hiddens,
                                           [loss, flatgrad] + cur_hiddens)
        self.f_loss = cgt.function([x_tnk, targ_tnk] + init_hiddens, loss)
        x_nk = cgt.matrix('x')
        outputs = self.net([x_nk] + init_hiddens)
        self.f_step = cgt.function([x_nk]+init_hiddens, outputs)

    def init_hidden_layers(self):
        return [np.zeros((self.size_batch, self.size_mem), cgt.floatX)
                for _ in xrange(self.num_hidden_layers)]


class CGTSolver(object):
    def __init__(self, cgt_net,
                 learning_rate=1.0, rho=0.9):
        self.cgt_net = cgt_net
        theta = self.cgt_net.params_col.get_value_flat()
        self.optim_state = rmsprop_create(theta, learning_rate, rho)
        self.num_epochs = 0

    def step(self, num_step, train_input, train_output):
        input_patches = self.process_data(train_input)
        output_patches = self.process_data(train_output)
        for _ in range(num_step):
            self.step_once(input_patches, output_patches)

    def step_once(self, input_patches, output_patches):
        self.num_epochs += 1
        print "starting epoch", self.num_epochs
        tstart = time.time()
        losses = []
        cur_hiddens = self.cgt_net.init_hidden_layers()
        for (x, y) in zip(input_patches, output_patches):
            assert x.shape[0] == y.shape[0] == self.cgt_net.n_unroll
            assert x.shape[1] == self.cgt_net.size_input
            assert y.shape[1] == self.cgt_net.size_output
            out = self.cgt_net.f_loss_and_grad(x, y, *cur_hiddens)
            loss, grad, cur_hiddens = out[0], out[1], out[2:]
            rmsprop_update(grad, self.optim_state)
            self.cgt_net.params_col.set_value_flat(self.optim_state.theta)
            losses.append(loss)
        print "%.3f s/batch. avg loss = %.3f" % \
              ((time.time() - tstart) / len(losses), np.mean(losses))
        self.optim_state.step_size *= .98

    def process_data(self, data):
        """
        Return a list of numpy arrays of shape (n_unroll, size_batch, size_*)
        If data is used for input, size_* = size_input
        """
        assert data.ndim == 2
        N = data.shape[0]
        size_patch = self.cgt_net.n_unroll * self.cgt_net.size_batch
        num_patches = N // size_patch
        patches = np.split(data[:num_patches * size_patch], num_patches, axis=0)
        patches = [
            np.reshape(patch, (self.cgt_net.n_unroll, self.cgt_net.size_batch, -1))
            for patch in patches
        ]
        return patches


def lstm(net_cnst, loss_func,
         n_layers, n_unroll,
         size_input, size_output, size_mem, size_batch):
    x_tnk = cgt.tensor3(name="X")
    targ_tnk = cgt.tensor3(name="T")
    num_hidden, net_ins_sym, net_outs_sym = \
        net_cnst(size_input, size_mem, n_layers, size_output, size_batch)
    net = nn.Module(net_ins_sym, net_outs_sym)

    loss = 0
    init_hiddens = [cgt.matrix() for _ in xrange(num_hidden)]
    cur_hiddens = init_hiddens
    for t in xrange(n_unroll):
        net_outs = net([x_tnk[t]] + cur_hiddens)
        cur_hiddens, net_out = net_outs[:-1], net_outs[-1]
        loss = loss + loss_func(net_out, targ_tnk[t])
    assert len(init_hiddens) == len(cur_hiddens)
    loss = loss / (n_unroll * size_batch)

    params = net.get_parameters()
    gradloss = cgt.grad(loss, params)
    flatgrad = cgt.concatenate([x.flatten() for x in gradloss])
    with utils.Message("compiling loss+grad"):
        f_loss_and_grad = cgt.function([x_tnk, targ_tnk] + init_hiddens,
                                       [loss, flatgrad] + cur_hiddens)
    f_loss = cgt.function([x_tnk, targ_tnk] + init_hiddens, loss)
    x_nk = cgt.matrix('x')
    outputs = net([x_nk] + init_hiddens)
    f_step = cgt.function([x_nk]+init_hiddens, outputs)
    return net, f_loss, f_loss_and_grad, f_step
