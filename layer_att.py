import tensorflow as tf
from keras.layers import Layer
from keras.activations import tanh, softmax

class BilinearAttentionLayer(Layer):
    def __init__(self, use_bias=True, **kwargs):
        super(BilinearAttentionLayer, self).__init__(**kwargs)
        self.use_bias = use_bias

    def build(self, input_shape):
        # Create trainable weight matrix W
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[1][-1], input_shape[1][-1]),
                                 initializer='glorot_uniform',
                                 trainable=True)
        if self.use_bias:
            # Create trainable bias vector b
            self.b = self.add_weight(name='b',
                                     shape=(1),
                                     initializer='zeros',
                                     trainable=True)
        super(BilinearAttentionLayer, self).build(input_shape)

    def call(self, inputs):
        input1, input2 = inputs
        # Perform the bilinear operation
        result = tf.matmul(input1, tf.transpose(self.W))
        result = tf.reduce_sum(result * input2, axis=-1, keepdims=True)
        if self.use_bias:
            result += self.b  # Add bias term if specified
        return softmax(tanh(result), axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0][0]