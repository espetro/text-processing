
import tensorflow as tf

class LittleModel:
    """Groups multiple utilities to transform a Tensorflow 2.x model into a TF Lite model,
    as well as to use it"""

    @staticmethod
    def run_tf_keras_model_in_tflite(from_bin=None, from_path=None):
        """TF Lite model can be loaded from an binary instance or from a .tflite file"""
        if from_bin is not None:
            interpreter = tf.lite.Interpreter(model_content=from_bin)
        elif from_path is not None:
            with open(from_path, "rb") as f:
                bin_model = f.read()

            interpreter = tf.lite.Interpreter(model_content=bin_model)

        # interpreter.allocate_tensors()
        # interpreter.invoke()
        return interpreter


    @staticmethod
    def store_tf_keras_model_in_tflite(from_keras=None, from_path=None, options: dict = None):
        """Keras model can be loaded from a Keras object instance or from a protobuf file. Options must be a dict.
        
        Options
        -------
            allow_custom_ops:
                bool, default True
            experimental_new_converter:
                bool, default True
            experimental_new_quantizer:
                bool, default True
            optimizations:
                List[tf.lite.Optimize], default [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
            supported_ops:
                List[tf.lite.OpsSet], default [tf.lite.OpsSet.SELECT_TF_OPS, tf.lite.OpsSet.TFLITE_BUILTINS]
            supported_types:
                List[tf.types], default [tf.float16]
        """
        if from_keras is not None:
            converter = tf.lite.TFLiteConverter.from_keras_model(from_keras)
        elif from_path is not None:
            converter = tf.lite.TFLiteConverter.from_saved_model(from_path)
        else:
            raise Exception("Parameters: ", from_keras, from_path)
            
        converter.allow_custom_ops = options.get("allow_custom_ops", True)
        converter.experimental_new_converter = options.get("experimental_new_converter", True)
        converter.experimental_new_quantizer = options.get("experimental_new_quantizer", True)

        converter.optimizations = options.get("optimizations", [tf.lite.Optimize.OPTIMIZE_FOR_SIZE])
        converter.target_spec.supported_ops = options.get("supported_ops", [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS])
        converter.target_spec.supported_types = options.get("supported_types", [tf.float16])
        
        return converter.convert()