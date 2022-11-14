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
# rabbitmqhost = "rabbitmq-clusterip-srv"
rabbitmqhost = "rabbitmq"
rabbitmqport = "5672"

_SIZE = 512


class TestDataSet(Dataset):
    def __len__(self):
        return _SIZE  # int(2**10)

    def __getitem__(self, i):
        body = "Ahem.. I think I'll be the only one who's saying this but yes, I was a lil bored during the film. Not to say that this is a bad movie, in fact it's a very good attempt at portraying the innermost emotions - dilemma, sorrow, love.., esp it's the director's debut (read from somewhere, is it true?). I felt that something's not quite right, maybe it's just me, I'm not drawn to the characters enough to immerse me in their world. This is a simple story, about ordinary people, ordinary lives. Through simple and short dialogs, the director tries to relate a simple guy's life, and how copes with the news of his illness by laughing it away every time. Oh ya his laughter was kinda cute at first but gradually it gets to me, such a deep hearty roar for a gentle man! I must say, I didn't feel the impact that most readers felt, in fact I was more drawn to the trivial scenarios like spitting of watermelon seeds with his sis that clearly shows that they're comfortable with each other, the granny who came back for another shot - this is kinda melancholic, the thoughtful gesture of writing down the procedures for his dad - hmm but this is predictable.. Don't misunderstood that I'm an action-lover, independent films are my cup of tea! Perhaps I just have a really high expectation after watching many deep films that have stronger imagery. Some Asian films worth the watch: <br /><br />Tony Takitani (depicts loneliness) Wayward Cloud (only 1 dialog) My Sassy Girl (I like it!) 4.30 (loneliness as well) 15 (gangsters lives in local setting) Before sunrise and Before sunset (I just have to mention these even though they are not Asian films. Fans will understand!)"
        return body


def calculate_pytorch(batch_sizes, number_of_tests, input_device):
    print("---- PYTORCH ----")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifierPytorch = pipeline(
        feature, model=model, tokenizer=tokenizer, device=input_device
    )

    test_pytorch = []
    for test in batch_sizes:
        test_pytorch.append(ResultsDataSet(test, [], []))

    for batch in test_pytorch:
        for i in range(number_of_tests):
            print("Batch_size: ", batch.size)
            dataset = TestDataSet()
            i = 0
            start = time.time()
            for res in classifierPytorch(
                dataset, truncation=True, batch_size=batch.size
            ):
                i = i + 1
                print(i)
            batch.test_time.append(time.time() - start)
            torch.cuda.empty_cache()
            batch.test_list.append(i)
            print("Number of Results: ", i)

    return test_pytorch


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
