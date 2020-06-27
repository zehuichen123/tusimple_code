import mxnet as mx
import numpy as np
import weight_softmax

num_roi = 100
num_class = 81

a = mx.sym.var(name = 'a', shape = (num_roi, num_class), dtype = 'float64')
b = mx.sym.var(name = 'b', shape = (num_roi, num_class), dtype = 'float64')

# c = mx.sym.broadcast_add(a, b, name = 'c')
# d = mx.sym.make_loss(mx.sym.broadcast_add(c, b, name = 'd'))
d = mx.sym.SoftmaxOutput(data=a, label=b)
d1 = mx.sym.Custom(
    data=a,
    label=b,
    op_type='weight_softmax'
)
d1 = mx.sym.make_loss(d1, name='d1')

data = mx.nd.random.normal(0, 1, (num_roi, num_class))
label = mx.nd.one_hot(mx.nd.random.randint(0, num_class, (num_roi,)), depth=num_class)

bind = d.simple_bind(ctx = mx.cpu(1))
bind1 = d1.simple_bind(ctx = mx.cpu(1))

bind.forward(a = data, b = label)
bind1.forward(a=data, b=label)

outputs = bind.outputs[0]
outputs1 = bind1.outputs[0]
# print(outputs)
bind.backward()
bind1.backward()
print(bind.grad_dict)
print('---')
print(bind1.grad_dict)
print("Diff")
print(np.sum(bind.grad_dict['a'] - bind1.grad_dict['a']))