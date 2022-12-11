# wykorzystaj docker image stworzony przez firmę Nvidia
FROM nvcr.io/nvidia/pytorch:22.07-py3

# pozwól na uruchamiania skyptów w koncie administratora dockera
ENV PIP_ROOT_USER_ACTION=ignore

# aktualizuj biblioteki
RUN apt-get update && apt-get install -y
RUN pip install --upgrade pip

# zainstaluj bibliotekę Transformers
RUN pip install fastai>=2.0.0 transformers tensorflow transformers[onnx]

# zainstaluj biblioteke dla RabbitMQ w Pythonie
RUN pip install pika --upgrade

# zainstaluj biblioteke do uruchomienia biblioteki ONNX z GPU
RUN pip install optimum[onnxruntime-gpu]

# skopiuj folder z przeprowadzonymi badaniami
COPY project /project
WORKDIR /../project