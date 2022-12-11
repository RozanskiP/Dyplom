import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from fastai.vision.all import *
from pathlib import Path
from torch.utils.data import Dataset
import numpy as np

from transformers import (
    AutoTokenizer,
    TFAutoModelForSequenceClassification,
    BertTokenizerFast,
    CamembertTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
    TFBertForSequenceClassification,
)
from optimum.onnxruntime import ORTModelForSequenceClassification

import tensorflow as tf

import os
import time
import pika
import json
import matplotlib.pyplot as plt

# to onnx
import transformers
from transformers.onnx import FeaturesManager

from results_data_set import ResultsDataSet

# type and name of model
feature = "text-classification"
model_name = "cmarkea/distilcamembert-base-sentiment"

# rabbitmq
QUEUE_NAME_ = "textClass"
# kubernetes or docker-compose
rabbitmqhost = "rabbitmq-clusterip-srv"
# rabbitmqhost = "rabbitmq"
rabbitmqport = "5672"

_SIZE = 2


class TestDataSet(Dataset):
    def __init__(self, channel):
        self.channel = channel

    def __len__(self):
        return _SIZE  # int(2**10)

    def __getitem__(self, i):
        method_frame, header_frame, body = self.channel.basic_get(QUEUE_NAME_, auto_ack=True)
        body_str = body.decode("utf-8")
        data = json.loads(body_str)
        self.time = data["time"]
        return data["name"]


def calculate_pytorch(classifierPytorch, channel):
    list_of_results = []

    dataset = TestDataSet(channel)
    for res in classifierPytorch(dataset, truncation=True):
        print(res)
        count_time = time.time() - dataset.time
        list_of_results.append(count_time)

    return list_of_results


def calculate_onnx(batch_sizes, number_of_tests, input_device):
    print("---- ONNX ----")
    model = ORTModelForSequenceClassification.from_pretrained(
        model_name, from_transformers=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifierONNX = pipeline(
        feature, model=model, tokenizer=tokenizer, device=input_device
    )

    test_onnx = []
    for batch in batch_sizes:
        test_onnx.append(ResultsDataSet(batch, [], []))

    for batch in test_onnx:
        for i in range(number_of_tests):
            print("Batch Size: ", batch.size)
            dataset = TestDataSet()
            i = 0
            start = time.time()
            for res in classifierONNX(dataset, truncation=True, batch_size=batch.size):
                i = i + 1
                print(i)
                # print(res)
            batch.test_time.append(time.time() - start)
            batch.test_list.append(i)
            print("Number of Results: ", i)

    return test_onnx


def calculate_tensorflow(batch_sizes, number_of_tests, input_device):
    print("---- TENSORFLOW ----")
    model = TFBertForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifierTensorflow = pipeline(
        feature, model=model, tokenizer=tokenizer, framework="tf", device=input_device
    )

    test_tensorflow = []
    for batch in batch_sizes:
        test_tensorflow.append(ResultsDataSet(batch, [], []))

    for batch in test_tensorflow:
        for i in range(number_of_tests):
            print("Batch Size: ", batch.size)
            dataset = TestDataSet()

            body = "Ahem.. I think I'll be the only one who's saying this but yes, I was a lil bored during the film. Not to say th"
            listBody = [body] * _SIZE

            arr = np.array(listBody).reshape(
                int(len(listBody) / batch.size), batch.size
            )
            arr = [list(x) for x in arr]

            tokens = tokenizer(arr[0], return_tensors="tf")
            model(tokens)

            start = time.time()
            for elem in arr:
                tokens = tokenizer(
                    elem, return_tensors="tf", padding="max_length", max_length=128
                )
                output = model(tokens)
                np.array(output)
                print(output.logits)

            batch.test_time.append(time.time() - start)
            batch.test_list.append(i)
            print("Number of Results: ", i)

    return test_tensorflow
