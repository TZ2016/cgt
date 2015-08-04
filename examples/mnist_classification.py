# Initially copied from https://github.com/Newmu/Theano-Tutorials/blob/master/4_modern_net.py
import cgt
from cgt import nn
from cgt.distributions import categorical
import numpy as np
from example_utils import fmt_row, fetch_dataset, train_test_val_slices
import time

def init_weights(*shape):
    return cgt.shared(np.random.randn(*shape) * 0.01)

def rmsprop_updates(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = cgt.grad(cost, params)
    updates = []
    for p, g in zip(params, grads):
        acc = cgt.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * cgt.square(g)
        gradient_scaling = cgt.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def model(X, w_h, w_h2, w_o, p_drop_input, p_drop_hidden, dropout):
    if dropout: X = nn.dropout(X, p_drop_input)
    h = nn.rectify(cgt.dot(X, w_h))

    if dropout: h = nn.dropout(h, p_drop_hidden)
    h2 = nn.rectify(cgt.dot(h, w_h2))

    if dropout: h2 = nn.dropout(h2, p_drop_hidden)
    py_x = nn.softmax(cgt.dot(h2, w_o))
    return h, h2, py_x

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--epochs",type=int,default=10)
    parser.add_argument("--profile",action="store_true")
    parser.add_argument("--dropout",action="store_true")
    args = parser.parse_args()

    mnist = fetch_dataset("http://rll.berkeley.edu/cgt/datasets/mnist.npz")

    Xdata = mnist["X"]/255.
    ydata = mnist["y"]

    # Xdata = Xdata[:100]
    # ydata = ydata[:100]

    np.random.seed(0)
    randperm = np.random.permutation(Xdata.shape[0])
    trainsli,testsli,valsli = train_test_val_slices(Xdata.shape[0],.8,.1,.1)
    traininds,testinds,valinds = randperm[trainsli],randperm[testsli],randperm[valsli]
    Xtrain = Xdata[traininds]
    ytrain = ydata[traininds]
    Xtest = Xdata[testinds]
    ytest = ydata[testinds]
    Xval = Xdata[valinds]
    yval = ydata[valinds]

    w_h = init_weights(784, 625)
    w_h2 = init_weights(625, 625)
    w_o = init_weights(625, 10)

    X = cgt.matrix("X")
    y = cgt.vector("y",dtype='i8')

    _h_drop, _h2_drop, pofy_drop = model(X, w_h, w_h2, w_o, 0.2, 0.5, args.dropout)
    _h_nodrop, _h2_nodrop, pofy_nodrop = model(X, w_h, w_h2, w_o, 0., 0., args.dropout)

    params = [w_h, w_h2, w_o]

    cost_drop = -cgt.mean(categorical.loglik(y, pofy_drop))
    updates = rmsprop_updates(cost_drop, params, lr=0.001)

    y_nodrop = cgt.argmax(pofy_drop, axis=1)
    cost_nodrop = -cgt.mean(categorical.loglik(y, pofy_nodrop))
    err_nodrop = cgt.cast(cgt.not_equal(y_nodrop, y), cgt.floatX).mean()

    train = cgt.function(inputs=[X, y], outputs=[], updates=updates)
    computeloss = cgt.function(inputs=[X, y], outputs=[err_nodrop,cost_nodrop])

    batch_size=64

    if args.profile: cgt.execution.profiler.start()

    print fmt_row(10, ["Epoch","Train NLL","Train Err","Test NLL","Test Err","Epoch Time"])
    for i_epoch in xrange(args.epochs):
        tstart = time.time()
        for start in xrange(0, Xtrain.shape[0], batch_size):
            end = start+batch_size
            train(Xtrain[start:end], ytrain[start:end])
        elapsed = time.time() - tstart
        trainerr, trainloss = computeloss(Xtrain, ytrain)
        testerr, testloss = computeloss(Xtest, ytest)
        print fmt_row(10, [i_epoch, trainloss, trainerr, testloss, testerr, elapsed])
    if args.profile: cgt.execution.profiler.print_stats()

if __name__ == "__main__":
    main()