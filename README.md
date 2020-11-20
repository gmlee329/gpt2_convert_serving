# gpt2_convert_serving

This repo aim to convert gpt2 pytorch model to tensorflow-serving  
This is the workflow : pytorch → onnx → tensorflow → tensorflow serving  

This repo referred to these references : https://github.com/onnx/onnx, https://github.com/onnx/models/tree/master/text/machine_comprehension/gpt-2, https://github.com/onnx/tutorials/blob/master/tutorials/OnnxTensorflowImport.ipynb

# How to use

**Run the docker container for conversion**

```bash 
$ git clone https://github.com/gmlee329/gpt2_convert_serving.git
$ docker build -t gpt_convert -f Dockerfile_conversion .
$ docker run --gpus 0 -it --name gpt_convert gpt_convert
```
Now, you are in docker container to convert pytorch to tensorflow

## pytorch -> onnx

In docker container, enter the command below

```bash 
$ python pytorch_to_onnx.py
```

Then, you can find the test_gpt2-lm-head folder  
In that folder, you can find the model.onnx file

## onnx -> tensorflow

Now, you have model.onnx file  
To convert onnx to tensorflow, enter the command below  

```bash 
$ python onnx_to_tensorflow.py
```

The model folder will be maded in the test_gpt2-lm-head folder  
In the model folder, there is assets, variables, saved_model.pb

## test converted tensorflow model's output

You can test converted tensorflow model with tf_model_test.py file  
Enter the command below

```bash 
$ python tf_model_test.py
```

## tensorflow -> tensorflow serving

You can also serving that converted tensorflow model 

**Run the docker container for serving**

Open another prompt window and enter the command below  
```bash 
$ docker build -t tf_serving -f Dockerfile_tf_serving .
$ docker run --gpus 0 -it -v 8501:8501 --name tf_serving tf_serving
```
(Port 8501 must need to serving)  
The tf_serving server will run on your local server

For test tf_serving, go back to gpt_conver container and enter the command below  

```bash 
$ python tf_serving_test.py
```

You can get result with vector form
Use that vector form result to generate text (refer to code in tf_model_test.py)
