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
    AutoModelForSequenceClassification,
    pipeline,
)
from optimum.onnxruntime import ORTModelForSequenceClassification
import onnxruntime as rt

import tensorflow as tf
from tensorflow.python.eager import context

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

_SIZE = 32


class TestDataSet(Dataset):
    def __init__(self, seq_size=1600):
        self.seq_size = seq_size

    def __len__(self):
        return _SIZE  # int(2**10)

    def __getitem__(self, i):
        # connection = pika.BlockingConnection(pika.ConnectionParameters(rabbitmqhost, rabbitmqport))
        # channel = connection.channel()
        # connection.close()
        # one long string to create same results
        body = "Ahem.. I think I'll be the only one who's saying this but yes, I was a lil bored during the film. Not to say that this is a bad movie, in fact it's a very good attempt at portraying the innermost emotions - dilemma, sorrow, love.., esp it's the director's debut (read from somewhere, is it true?). I felt that something's not quite right, maybe it's just me, I'm not drawn to the characters enough to immerse me in their world. This is a simple story, about ordinary people, ordinary lives. Through simple and short dialogs, the director tries to relate a simple guy's life, and how copes with the news of his illness by laughing it away every time. Oh ya his laughter was kinda cute at first but gradually it gets to me, such a deep hearty roar for a gentle man! I must say, I didn't feel the impact that most readers felt, in fact I was more drawn to the trivial scenarios like spitting of watermelon seeds with his sis that clearly shows that they're comfortable with each other, the granny who came back for another shot - this is kinda melancholic, the thoughtful gesture of writing down the procedures for his dad - hmm but this is predictable.. Don't misunderstood that I'm an action-lover, independent films are my cup of tea! Perhaps I just have a really high expectation after watching many deep films that have stronger imagery. Some Asian films worth the watch: <br /><br />Tony Takitani (depicts loneliness) Wayward Cloud (only 1 dialog) My Sassy Girl (I like it!) 4.30 (loneliness as well) 15 (gangsters lives in local setting) Before sunrise and Before sunset (I just have to mention these even though they are not Asian films. Fans will understand!)"
        return body[: self.seq_size]


def calculate_pytorch(
    number_of_cores,
    number_of_tests,
    input_device,
    useCores=True,
    changingBatch=True,
    set_seq_size=None,
):
    print("---- PYTORCH ----")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifierPytorch = pipeline(
        feature, model=model, tokenizer=tokenizer, device=input_device
    )

    test_pytorch = []
    for test in number_of_cores:
        test_pytorch.append(ResultsDataSet(test, [], []))

    for core in test_pytorch:
        for i in range(number_of_tests):
            print("Cores: ", core.size)
            if useCores:
                torch.set_num_threads(core.size)
            else:
                torch.set_num_threads(1)
            if set_seq_size != None:
                dataset = TestDataSet()
            else:
                dataset = TestDataSet(set_seq_size)
            i = 0
            start = time.time()
            if changingBatch:
                for res in classifierPytorch(
                    dataset, truncation=True, batch_size=core.size
                ):
                    i = i + 1
                    print(i)
            else:
                for res in classifierPytorch(dataset, truncation=True, batch_size=16):
                    i = i + 1
                    print(i)
            core.test_time.append(time.time() - start)
            core.test_list.append(i)
            print("Number of Results: ", i)

    return test_pytorch


def calculate_onnx(number_of_cores, number_of_tests, input_device):
    print("---- ONNX ----")
    model = ORTModelForSequenceClassification.from_pretrained(
        model_name, from_transformers=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifierONNX = pipeline(
        feature, model=model, tokenizer=tokenizer, device=input_device
    )

    sess_options = rt.SessionOptions()
    sess_options.enable_profiling = True
    sess_options.intra_op_num_threads = 2
    sess_options.inter_op_num_threads = 2
    sess_options.execution_mode = rt.ExecutionMode.ORT_PARALLEL
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL

    test_onnx = []
    for test in number_of_cores:
        test_onnx.append(ResultsDataSet(test, [], []))

    for core in test_onnx:
        for i in range(number_of_tests):
            print("Cores: ", core.size)
            torch.set_num_threads(core.size)
            dataset = TestDataSet()
            i = 0
            start = time.time()
            for res in classifierONNX(dataset, truncation=True, batch_size=core.size):
                i = i + 1
                print(i)
            core.test_time.append(time.time() - start)
            core.test_list.append(i)
            print("Number of Results: ", i)

    return test_onnx


def calculate_tensorflow(number_of_cores, number_of_tests, input_device):
    print("---- TENSORFLOW ----")
    model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifierTensorflow = pipeline(
        feature, model=model, tokenizer=tokenizer, framework="tf", device=input_device
    )

    test_tensorflow = []
    for test in number_of_cores:
        test_tensorflow.append(ResultsDataSet(test, [], []))

    for core in test_tensorflow:
        for i in range(number_of_tests):
            print("Cores: ", core.size)

            _ = tf.Variable([1])
            context._context = None
            context._create_context()
            tf.config.threading.set_inter_op_parallelism_threads(2)

            body = "Ahem.. I think I'll be the only one who's saying this but yes, I was a lil bored during the film. Not to say th"
            listBody = [body] * _SIZE

            arr = np.array(listBody).reshape(int(len(listBody) / core.size), core.size)
            arr = [list(x) for x in arr]

            dataset = TestDataSet()
            i = 0
            start = time.time()

            for elem in arr:
                tokens = tokenizer(elem, return_tensors="tf")
                output = model(tokens)
                np.array(output)
                print(output.logits)

            core.test_time.append(time.time() - start)
            core.test_list.append(i)
            print("Number of Results: ", i)

    return test_tensorflow
