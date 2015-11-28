import cgt
from cgt import nn


# def mux_layer(X, size_in, size_out, splits, funcs):
#     if not isinstance(funcs, (list, tuple)):
#         funcs = [funcs] * (len(splits) + 1)
#     assert len(splits) + 1 == len(funcs) > 0
#     splits.append(size_out)
#     assert all(splits[i] < splits[i+1] for i in xrange(len(splits) - 1))
#     curr, outs = 0, []
#     for _s, _f in zip(splits, funcs):
#         _o = _f(X, size_in, _s - curr)
#         outs.append(_o)
#         curr = _s
#     return outs
#
#
# def demux_layer(Xs, funcs):
#     assert isinstance(Xs, list)
#     assert len(funcs) == len(Xs)
#     curr, outs = 0, []
#     for _X, _f in zip(Xs, funcs):
#         _o = _f(_X) if _f is not None else _X
#         outs.append(_o)
#     out = cgt.concatenate(outs, axis=1)
#     return out


def combo_layer(X, size_in, size_out, splits, funcs, name='?', dbg_out={}):
    assert isinstance(splits, tuple) and len(splits) > 0
    assert all(splits[i] < splits[i+1] for i in xrange(len(splits) - 1))
    assert splits[0] >= 0 and splits[-1] <= size_out
    splits = list(splits)
    assert not isinstance(funcs, list)
    if not isinstance(funcs, tuple):
        funcs = [funcs] * (len(splits) + 1)
    else:
        funcs = list(funcs)
    assert len(splits) + 1 == len(funcs)
    if splits[0] == 0:
        splits.pop(0)
        funcs.pop(0)
    if len(splits) > 0 and splits[-1] == size_out:
        splits.pop()
        funcs.pop()
    curr, names, ins, outs = 0, [], [], []
    splits.append(size_out)
    for _split, _func in zip(splits, funcs):
        _name = 'L%s[%d:%d]' % (name, curr, _split)
        _i = cgt.sigmoid(nn.Affine(size_in, _split - curr, name=_name)(X))
        _o = _func(_i) if _func is not None else _i
        curr = _split
        ins.append(_i)
        outs.append(_o)
        names.append(_name)
    out = cgt.concatenate(outs, axis=1) if len(outs) > 1 else outs[0]
    dbg_out.update(dict(zip([_n + '~in' for _n in names], ins)))
    dbg_out.update(dict(zip([_n + '~out' for _n in names], outs)))
    return out
