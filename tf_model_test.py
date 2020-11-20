import numpy as np
import tensorflow as tf
import torch
from transformers import GPT2Tokenizer
from torch.nn import functional as F

def to_tf_tensor(torch_tensor):
    np_tensor = torch_tensor.numpy()
    tf_tensor = tf.convert_to_tensor(np_tensor)
    return tf_tensor
def to_torch_tensor(tf_tensor):
    np_tensor = tf_tensor.numpy()
    torch_tensor = torch.tensor(np_tensor)
    return torch_tensor

tf_model = tf.saved_model.load("./test_gpt2-lm-head/model").signatures["serving_default"]
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

input_torch = tokenizer.encode("My name is Thomas and my main", add_special_tokens=True, return_tensors='pt')
input_tensor = to_tf_tensor(input_torch)

output = input_tensor

for i in range(length):
    outputs = tf_model(output)
    
    logits = outputs['output_0']
    logits = to_torch_tensor(logits)
    logits = logits[:, -1, :]

    log_probs = F.softmax(logits, dim=-1)
    _, prev = torch.topk(log_probs, k=1, dim=-1)
    
    output = to_torch_tensor(output)
    output = torch.cat((output, prev), dim=1)
    output = to_tf_tensor(output)

output = output.numpy().tolist()[0]
text = tokenizer.decode(output)
print(text)