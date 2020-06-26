from mxnet.gluon.nn import HybridBlock, HybridSequential
from mxnet.gluon.nn import Conv2D, Flatten, Dense, Dropout
from mxnet.gluon.loss import L2Loss
from mxnet.gluon.model_zoo.vision import resnet34_v1

import importlib_resources as pkg_resources
import mxnet as mx

class SegmentationNetwork(HybridBlock):
    """Paragraph segmentation network based on Deep CNN.

    Source:
        https://github.com/awslabs/handwritten-text-recognition-for-apache-mxnet
    """
    MODEL_PATH = pkg_resources.files("preproc.data").joinpath("cnn_mse.params")

    def __init__(self, p_dropout=0.5, ctx=mx.cpu()):
        super(SegmentationNetwork, self).__init__()

        self.ctx = ctx
        self.cnn = self.load_model(p_dropout)
        
        self.hybridize()
        self.collect_params().reset_ctx(self.ctx)

    def load_model(self, p_dropout):
        pretrained = resnet34_v1(pretrained=True, ctx=self.ctx)
        first_weights = pretrained.features[0].weight.data().mean(axis=1).expand_dims(axis=1)

        body = HybridSequential(prefix="ParagraphSegmentation_")
        with body.name_scope():
            first_layer = Conv2D(
                channels=64, kernel_size=(7, 7), padding=(3, 3), strides=(2, 2), in_channels=1, use_bias=False
            )
            first_layer.initialize(mx.init.Normal(), ctx=self.ctx)
            first_layer.weight.set_data(first_weights)

            body.add(first_layer)
            body.add(*pretrained.features[1:6])
        
            output = HybridSequential()
            with output.name_scope():
                output.add(Flatten())
                output.add(Dense(64, activation='relu'))
                output.add(Dropout(p_dropout))
                output.add(Dense(64, activation='relu'))
                output.add(Dropout(p_dropout))
                output.add(Dense(4, activation='sigmoid'))

            output.collect_params().initialize(mx.init.Normal(), ctx=self.ctx)
            body.add(output)
        
        return body

    def hybrid_forward(self, F, x):
        return self.cnn(x)

    def load_params(self, params_fpath=None):
        params_fpath = params_fpath or str(SegmentationNetwork.MODEL_PATH)
        self.cnn.load_parameters(params_fpath)



if __name__ == "__main__":
    net = SegmentationNetwork()
    net.load_params()
    

