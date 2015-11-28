# Learning Stochastic Feedforward Neural Networks

import os
import cgt
import traceback
import pprint
from cgt.core import get_surrogate_func, Node
from cgt import nn
import numpy as np
import pickle
from cgt.utility.param_collection import ParamCollection
from cgt.distributions import gaussian_diagonal

from opt import rmsprop_create, rmsprop_update, adam_create, adam_update
from debug import example_debug, safe_path
from layers import combo_layer


def err_handler(type, flag):
    print type, flag
    traceback.print_stack()
    raise FloatingPointError('refer to err_handler for more details')
np.seterr(divide='call', over='warn', invalid='call', under='warn')
np.seterrcall(err_handler)
np.set_printoptions(precision=4, suppress=True)
print cgt.get_config(True)
cgt.check_source()


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


def mask_layer(func, X, size_in, i_start, i_end=None):
    if i_end is None:
        i_start, i_end = 0, i_start
    assert isinstance(i_start, int) and isinstance(i_end, int)
    assert -1 < i_start <= i_end <= size_in
    if i_end == i_start:
        return X
    if i_end - i_start == size_in:
        return func(X)
    outs = []
    if i_start > 0:
        outs.append(X[:, :i_start])
    outs.append(func(X[:, i_start:i_end]))
    if i_end < size_in:
        outs.append(X[:, i_end:])
    out = cgt.concatenate(outs, axis=1)
    return out


def lstm_layers(size_in, size_out, num_units):
    """
    Construct a recurrent neural network with multiple layers of LSTM units,
    with each layer a block of cells sharing a common set of gate units.
    Return list of inputs and a list of outputs for the net at one time step.
    Inputs =  [ net_in, hidden layers ]
    Outputs = [ hidden layers, net_out ]

    :param size_in: input dimension
    :param num_units: number of memory units for each layer
    :param size_out: output dimension
    :return:
    :rtype: (int, list, list)
    """
    net_in = cgt.matrix("X", fixed_shape=(None, size_in))
    net_c_prev, net_h_prev = [], []
    net_c_curr, net_h_curr = [], []
    prev_l_num_units, prev_out = size_in, net_in
    for l_num_units in num_units:
        c_prev = cgt.matrix(fixed_shape=(None, l_num_units))
        h_prev = cgt.matrix(fixed_shape=(None, l_num_units))
        c_curr, h_curr = lstm_block(h_prev, c_prev, prev_out,
                                    prev_l_num_units, l_num_units)
        net_c_prev.append(c_prev)
        net_h_prev.append(h_prev)
        net_c_curr.append(c_curr)
        net_h_curr.append(h_curr)
        prev_l_num_units = l_num_units
        prev_out = h_curr
    inputs = [net_in] + net_c_prev + net_h_prev
    outputs = net_c_curr + net_h_curr
    return inputs, outputs


def hybrid_network(size_in, size_out, num_units, num_stos, dbg_out={}):
    assert len(num_units) == len(num_stos)
    net_in = cgt.matrix("X", fixed_shape=(None, size_in))
    prev_num_units, prev_out = size_in, net_in
    dbg_out['NET~in'] = net_in
    curr_layer = 1
    for (curr_num_units, curr_num_sto) in zip(num_units, num_stos):
        assert curr_num_units >= curr_num_sto >= 0
        prev_out = combo_layer(prev_out, prev_num_units, curr_num_units,
                  (curr_num_sto,),
                  (cgt.bernoulli, None),
                  name=str(curr_layer), dbg_out=dbg_out)
        dbg_out['L%d~out' % curr_layer] = prev_out
        prev_num_units = curr_num_units
        curr_layer += 1
    net_out = nn.Affine(prev_num_units, size_out,
                        name="InnerProd(%d->%d)" % (prev_num_units, size_out)
                        )(prev_out)
    dbg_out['NET~out'] = net_out
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
    if config['variance'] == 'in':
        print "Input includes diagonal variance"
        # TODO_TZ diagonal for now
        in_var = cgt.matrix('V', fixed_shape=(None, config['num_inputs']))
        inputs.append(in_var)
        out_mean, out_var = net_out, in_var
    elif config['variance'] == 'out':  # net outputs variance
        print "Network outputs diagonal variance"
        cutoff = size_out // 2
        out_mean, out_var = net_out[:, :cutoff], net_out[:, cutoff:]
        # out_var = out_var ** 2 + 1.e-6
        out_var = cgt.exp(out_var) + 1.e-6
    else:
        print "Constant variance"
        assert isinstance(config['variance'], float)
        out_mean = net_out
        out_var = cgt.fill(config['variance'], [size_batch, size_out])
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
    f_surr = get_surrogate_func(inputs + [Y], net_out,
                                [loss_raw], params, _dbg_out=dbg_out)
    # TODO_TZ f_step seems not to fail if X has wrong dim
    return params, f_step, None, None, f_surr


def step(X, Y, workspace, config, Y_var=None, dbg_iter=None, dbg_done=None):
    if config['debug'] and (dbg_iter is None or dbg_done is None):
        dbg_iter, dbg_done = example_debug(config, X, Y, Y_var=Y_var)
    if config['variance'] == 'in': assert Y_var is not None
    f_surr, f_step = workspace['f_surr'], workspace['f_step']
    param_col = workspace['param_col']
    optim_state = workspace['optim_state']
    num_epochs = num_iters = 0
    while num_epochs < config['n_epochs']:
        ind = np.random.choice(X.shape[0], config['size_batch'])
        # ind = [num_iters]  # for ordered data
        x, y = X[ind], Y[ind]
        if config['variance'] == 'in':
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
            # TODO remove the below
            h_prob = np.exp(info['objective_unweighted'] - info['weights_raw_log'])
            print np.unique(np.round(h_prob, 2), return_counts=True)
            print np.unique(np.round(info['weights'], 3), return_counts=True)
        if dbg_iter:
            dbg_iter(num_epochs, num_iters, info, workspace)
    # save params
    if 'dump_path' in config:
        out_path = config['dump_path']
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        print "Saving params to %s" % out_path
        # pickle.dump(args, open(_safe_path('args.pkl'), 'w'))
        pickle.dump(param_col.get_values(), safe_path('params.pkl', out_path, 'w'))
        pickle.dump(optim_state, safe_path('__snapshot.pkl', out_path, 'w'))
    if dbg_done: dbg_done(workspace)
    return param_col, optim_state


def create(args):
    dbg_out = {}
    net_in, net_out = hybrid_network(args['num_inputs'], args['num_outputs'],
                                     args['num_units'], args['num_sto'],
                                     dbg_out=dbg_out)
    if not args['dbg_out_full']: dbg_out = {}
    params, f_step, f_loss, f_grad, f_surr = \
        make_funcs(net_in, net_out, args, dbg_out=dbg_out)
    param_col = ParamCollection(params)
    if 'snapshot' in args:
        print "Loading params from previous snapshot: %s" % args['snapshot']
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
        print "Initializing theta from fresh"
        if method == 'rmsprop':
            optim_state = rmsprop_create(theta, step_size=args['step_size'])
            f_update = rmsprop_update
        elif method == 'adam':
            optim_state = adam_create(theta, step_size=args['step_size'])
            f_update = adam_update
        else:
            raise ValueError('unknown optimization method: %s' % method)
        init_method = args['init_theta']['distr']
        if init_method == 'XavierNormal':
            init_theta = nn.XavierNormal(**args['init_theta']['params'])
        elif init_method == 'gaussian':
            init_theta = nn.IIDGaussian(**args['init_theta']['params'])
        else:
            raise ValueError('unknown init distribution')
        optim_state['theta'] = nn.init_array(
            init_theta, (param_col.get_total_size(), 1)).flatten()
    param_col.set_value_flat(optim_state['theta'])
    workspace = {
        'optim_state': optim_state,
        'param_col': param_col,
        'f_surr': f_surr,
        'f_step': f_step,
        'f_loss': f_loss,
        'f_grad': f_grad,
        'update': f_update,
    }
    print "Configurations"
    pprint.pprint(args)
    return workspace


if __name__ == "__main__":
    import yaml
    import time
    from data import scale_data, data_synthetic_a

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
