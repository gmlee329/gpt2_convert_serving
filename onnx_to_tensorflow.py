import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

model_onnx = onnx.load('./test_gpt2-lm-head/model.onnx')
tf_rep = prepare(model_onnx)

tf_rep.export_graph('./test_gpt2-lm-head/model')