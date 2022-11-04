FROM nvcr.io/nvidia/pytorch:22.07-py3

# set possibility to install command in root
ENV PIP_ROOT_USER_ACTION=ignore

# upgrade libraries
RUN apt-get update && apt-get install -y
RUN pip install --upgrade pip

# install default library for transformers api
RUN pip install fastai>=2.0.0 transformers tensorflow transformers[onnx]

# rabbitmq for python
RUN pip install pika --upgrade

# transformers for onnx with gpu
RUN pip install optimum[onnxruntime-gpu]

