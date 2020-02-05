import tensorflow as tf
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
import numpy as np


#Swish function has been found useful in vanilla dense networks.
#Works well in this case. https://arxiv.org/pdf/1710.05941v1.pdf
def swish(x, beta = 1.0):
    return (x * tf.keras.activations.sigmoid(beta * x))

#Implements standard scaling as the first layer of the network.
#I.e. for each input it subracts the mean and divides by the standard deviation
#The mean and std (=scale) are calculated from training set and used as input when
#defining this layer.
class StandardScalerLayer(tf.keras.layers.Layer):
    def __init__(self, means, scale, **kwargs):
        self.means = np.array(means)
        self.scale = np.array(scale)
        self.tensorMeans = tf.convert_to_tensor(np.reshape(self.means, (1, self.means.shape[-1])), dtype='float32')
        self.invertedScale = tf.convert_to_tensor(1.0 / np.reshape(self.scale, (1, self.scale.shape[-1])), dtype='float32')
        super(StandardScalerLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(StandardScalerLayer, self).build(input_shape)

    def call(self, input):
        return tf.math.multiply((input - self.tensorMeans), self.invertedScale)

    def get_config(self):
        return {'means': self.means, 'scale': self.scale}

#Convenience function to create the classifier network.
#makes playing around with hyperparameters easier.
def createClassifier(nInputs, means, scales):
    _initializer = "glorot_normal"
    _regularizer = tf.keras.regularizers.l2(0.0) #(1e-4)
    _activation = swish
    _neurons = 64
    _blocks = 5
    _rate = 0.1
    inputs = tf.keras.Input(shape=(nInputs), name="classifierInput")
    x = StandardScalerLayer(means, scales, name="Input")(inputs)
    for i in range(_blocks):
        x = Dense(_neurons, activation=_activation, kernel_initializer=_initializer, kernel_regularizer=_regularizer)(x)
        x = BatchNormalization()(x)
        x = Dropout(_rate)(x)
    outputs = Dense(1, activation="sigmoid", name="classifierOutput")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="classifier")

    return model

#Creates frozen_graph.pb which can be uploaded to the CMSSW for inference.
#Complies with the PhysicsTools/TensorFlow C++ API.
#The input node gets the name 'x' and output 'Identity'. These are needed
#in the C++ APIs tensorflow::run() function.

def createFrozenModel(model):
    full_model = tf.function(lambda x: model(x))
    full_model = full_model.get_concrete_function(
        tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
    )

    from tensorflow.python.compiler.tensorrt import trt_convert as trt
    frozen_func = trt.convert_to_constants.convert_variables_to_constants_v2(full_model)
    frozen_func.graph.as_graph_def()

    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./frozen_models",
                      name="frozen_graph.pb",
                      as_text=False)

    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    for layer in layers:
        print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)

    print("Frozen model outputs: ")
    print(frozen_func.outputs)
    print("Graph frozen")