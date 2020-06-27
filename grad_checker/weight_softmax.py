import numpy as np
import mxnet as mx
import time

class WeightSoftmaxOperator(mx.operator.CustomOp):
    def __init__(self):
        super().__init__()
    
    def forward(self, is_train, req, in_data, out_data, aux):
        data = in_data[0]
        label = in_data[1]

        data = mx.nd.softmax(data, axis=-1)
        loss = - label * mx.nd.log(data + 1e-8)
        self.assign(out_data[0], req[0], loss)
    
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        data = in_data[0]
        label = in_data[1]

        d_grad = mx.nd.softmax(data, axis=-1) - label
        
        self.assign(in_grad[0], req[0], d_grad)
        self.assign(in_grad[1], req[1], mx.nd.zeros_like(label))
        

@mx.operator.register('weight_softmax')
class WeightSoftmaxProp(mx.operator.CustomOpProp):
    def __init__(self):
        super().__init__(need_top_grad=True)

    def list_arguments(self):
        return ['data', 'label']
    
    def list_outputs(self):
        return ['output']
    
    def infer_shape(self, in_shape):
        return [in_shape[0], in_shape[1]], [in_shape[0]]
    
    def create_operator(self, ctx, shapes, dtypes):
        return WeightSoftmaxOperator()
    
    # def declare_backward_dependency(self, out_grad, in_data, out_data):
    #     deps = []
    #     if self.need_top_grad_:
    #         deps.extend(out_grad)
    #     deps.extend(in_data)
    #     return deps


