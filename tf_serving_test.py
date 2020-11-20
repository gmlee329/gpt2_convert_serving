import json
import numpy as np
import tensorflow as tf
from transformers import GPT2Tokenizer
import requests

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def keras_model_predict(prompt):
    URL = "https://localhost:8501/v1/models/keras:predict"

    input_torch = tokenizer.encode(prompt, add_special_tokens=True, return_tensors='pt')
    np_tensor = input_torch.numpy()
    input_tensor = tf.convert_to_tensor(np_tensor)
    input_tensor = input_tensor.numpy().tolist()

    data = { "signature_name" : "serving_default", 
    "inputs": { "input1" : input_tensor }
    }
    data = json.dumps(data)

    response = requests.post(URL, data=data)

    if response.status_code == 200:
        text = response.text 
        text = json.loads(text) # 결과가 {'outputs': }의 꼴로 나온다.
        
        prediction = text['outputs']
        
        generation = np.array(prediction['output_0'])  # signature에서 설정한 이름으로 꺼낸다.
        
        return generation
    else:
        print("Failed")
        return False

prompt = "My name is Thomas and my main"
output = keras_model_predict(prompt)