import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, AlphaDropout
import numpy as np

from tensorflow.keras.activations import selu

#Swish function has been found useful in vanilla dense networks.
#Works well in this case. https://arxiv.org/pdf/1710.05941v1.pdf
def swish(x, beta = 1.0):
    return (x * tf.keras.activations.sigmoid(beta * x))

#Activation function with self normalizing properties.
scale = 1.07862
alpha = 2.90427
def serlu(x):
    return scale * tf.where(x >= 0.0, x, alpha * x * tf.exp(x))

#Sanitizing layer that combines the clamping and standard scaling. Should be the first layer after Input.
#expectes minValues, maxValues, means, scale to be ndarrays
class InputSanitizerLayer(tf.keras.layers.Layer):
    def __init__(self, means, scale, minValues, maxValues, **kwargs):
        self.minValues = minValues
        self.maxValues = maxValues
        self.means = means
        self.scale = scale

        self.tensorMeans = tf.convert_to_tensor(np.reshape(self.means, (1, self.means.shape[-1])), dtype='float32')
        self.invertedScale = tf.convert_to_tensor(1.0 / np.reshape(self.scale, (1, self.scale.shape[-1])), dtype='float32')
        self.tensorMins = tf.convert_to_tensor(np.reshape(self.minValues, (1, self.minValues.shape[-1])), dtype='float32')
        self.tensorMaxs = tf.convert_to_tensor(np.reshape(self.maxValues, (1, self.maxValues.shape[-1])), dtype='float32')
        super(InputSanitizerLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(InputSanitizerLayer, self).build(input_shape)

    def call(self, input):
        return tf.math.multiply((tf.math.maximum(tf.math.minimum(input, self.tensorMaxs), self.tensorMins)-self.tensorMeans), self.invertedScale)

    def get_config(self):
        return {'means': self.means, 'scale': self.scale, 'minValues': self.minValues, 'maxValues': self.maxValues}


#Layer that clamps the input values given to the network. Simplifies
#deploying the network as it takes care of the clamping without need
#to manually configure this to the framework (CMSSW)

# Assuming
# minValues = np.array()
# maxValues = np.array()
class ClampLayer(tf.keras.layers.Layer):
    def __init__(self, minValues, maxValues, **kwargs):
        self.minValues = minValues
        self.maxValues = maxValues
        self.tensorMins = tf.convert_to_tensor(np.reshape(self.minValues, (1, self.minValues.shape[-1])), dtype='float32')
        self.tensorMaxs = tf.convert_to_tensor(np.reshape(self.maxValues, (1, self.maxValues.shape[-1])), dtype='float32')
        super(ClampLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ClampLayer, self).build(input_shape)

    def call(self, input):
        return tf.math.maximum(tf.math.minimum(input, self.tensorMaxs), self.tensorMins)

    def get_config(self):
        return {'minValues': self.minValues, 'maxValues': self.maxValues}

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
        tf.math.multiply((input-self.tensorMeans), self.invertedScale)

    def get_config(self):
        return {'means': self.means, 'scale': self.scale}

#Convenience function to create the classifier network.
#makes playing around with hyperparameters easier.
def createClassifier(nInputs, means, scales, minValues, maxValues):
    _initializer = "lecun_normal"
    _activation = serlu
    _neurons = 64
    _blocks = 10
    _rate = 0.1
    _rateRegularizer = 2e-5
    inputs = tf.keras.Input(shape=(nInputs), name="classifierInput")
    # x = InputSanitizerLayer(means, scales, minValues, maxValues)(inputs)
    for i in range(_blocks):
#        x = Dense(_neurons, activation=_activation, kernel_initializer=_initializer, activity_regularizer=tf.keras.regularizers.l1_l2(_rateRegularizer))(x)
        x = Dense(_neurons, activation=_activation, kernel_initializer=_initializer, activity_regularizer=tf.keras.regularizers.l1_l2(_rateRegularizer))(inputs)

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
