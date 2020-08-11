from tensorflow.keras.layers import Conv2D, Activation, Multiply

class FullGatedConv2D(Conv2D):
    """Gated Convolutional Class"""

    def __init__(self, filters=32, **kwargs):
        super().__init__(filters=filters * 2, **kwargs)
        self.gated_filters = filters

    def call(self, inputs):
        """Apply gated convolution"""
        output = super().call(inputs)
        linear = Activation("linear")(output[:, :, :, :self.gated_filters])
        sigmoid = Activation("sigmoid")(output[:, :, :, self.gated_filters:])

        return Multiply()([linear, sigmoid])

    def compute_output_shape(self, input_shape):
        """Compute shape of layer output"""
        output_shape = super().compute_output_shape(input_shape)
        return tuple(output_shape[:3]) + (self.gated_filters,)

    def get_config(self):
        """Return the config of the layer"""
        config = super().get_config()
        config.update({"gated_filters": self.gated_filters})
        return config

