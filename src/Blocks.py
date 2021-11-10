import tensorflow as tf
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model

regul = 0.001


class RelationalModel(Layer):
    def __init__(self, input_size, n_of_features, filters, rm=None, reuse_model=False, **kwargs):
        self.input_size = input_size
        self.n_of_features = n_of_features
        self.filters = filters
        n_of_filters = len(filters)
        if (reuse_model):
            relnet = rm
        else:
            input1 = Input(shape=(n_of_features,))
            x = input1
            for i in range(n_of_filters - 1):
                x = Dense(filters[i], kernel_regularizer=regularizers.l2(regul),
                          activity_regularizer=regularizers.l2(regul),
                          bias_regularizer=regularizers.l2(regul), activation='relu')(x)

            x = Dense(filters[-1], kernel_regularizer=regularizers.l2(regul),
                      bias_regularizer=regularizers.l2(regul), activation='linear')(x)
            relnet = Model(inputs=[input1], outputs=[x])
        self.relnet = relnet
        self.output_size = filters[-1]

        super(RelationalModel, self).__init__(**kwargs)

    def build(self, input_shape):
        self.relnet.build((None, self.n_of_features))  # ,self.input_size
        # self.weights = self.relnet.weights

    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        input_size = self.input_size
        return (None,) + input_size + (int(output_size),)

    def call(self, X):
        X = tf.reshape(X, (-1, self.n_of_features))
        output = self.relnet.call(X)
        output = tf.reshape(output, ((-1,) + self.input_size + (self.output_size,)))
        return output

    def getRelnet(self):
        return self.relnet

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'input_size':    self.input_size,
            'n_of_features': self.n_of_features,
            'filters':       self.filters,
        })
        return config


class ObjectModel(Layer):
    def __init__(self, input_size, n_of_features, filters, om=None, reuse_model=False, **kwargs):
        self.input_size = input_size
        self.n_of_features = n_of_features
        self.filters = filters
        n_of_filters = len(filters)

        if (reuse_model):
            objnet = om
        else:
            input1 = Input(shape=(n_of_features,))
            x = input1
            for i in range(n_of_filters - 1):
                x = Dense(filters[i], kernel_regularizer=regularizers.l2(regul),
                          activity_regularizer=regularizers.l2(regul),
                          bias_regularizer=regularizers.l2(regul), activation='relu')(x)
            x = Dense(filters[-1], kernel_regularizer=regularizers.l2(regul),
                      bias_regularizer=regularizers.l2(regul), activation='linear')(x)

            objnet = Model(inputs=[input1], outputs=[x])
        self.objnet = objnet

        self.output_size = filters[-1]
        super(ObjectModel, self).__init__(**kwargs)

    def build(self, input_shape):
        self.objnet.build((None, self.input_size, self.n_of_features))
        # self.weights = self.objnet.weights

    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        input_size = self.input_size

        return (None,) + input_size + (int(output_size),)

    def call(self, X):
        X = tf.reshape(X, (-1, self.n_of_features))
        output = self.objnet.call(X)
        output = tf.reshape(output, ((-1,) + self.input_size + (self.output_size,)))

        return output

    def getObjnet(self):
        return self.objnet

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'input_size':    self.input_size,
            'n_of_features': self.n_of_features,
            'filters':       self.filters,
        })
        return config
