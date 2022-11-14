from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import time
import torch
import matplotlib.pyplot as plt
from optimum.onnxruntime import ORTModelForSequenceClassification
import os

feature = "text-classification"
model_name = "cmarkea/distilcamembert-base-sentiment"

_SIZE = 256


def average(arr):
    return sum(arr) / len(arr)


def cal_results(list_test):
    x = []
    y = []
    for index, batch in enumerate(list_test):
        x.append(batch.size)
        y.append(average(batch.test_time))
    return x, y


class TestDataSet(Dataset):
    def __len__(self):
        return _SIZE

    def __getitem__(self, i):
        body = "I was a lil bored during the film. Not to say that this is a bad movie."
        body = "I was a lil bored during the film."
        return body


class ResultsDataSet:
    def __init__(self, size, test_list, test_time):
        self.size = size
        self.test_list = test_list
        self.test_time = test_time


def savedata(file_name, data):
    datax, datay = cal_results(data)

    file_w = open(f"{file_name}.txt", "w")

    for value in datax:
        file_w.write(f"{value}, ")
    file_w.write(f"\n")
    for value in datay:
        file_w.write(f"{value}, ")
    file_w.close()


def run():

    batch_sizes = list(range(1, 64, 3))
    test_pytorch = []
    for test in batch_sizes:
        test_pytorch.append(ResultsDataSet(test, [], []))
    number_of_tests = 10

    test_onnx = []
    for test in batch_sizes:
        test_onnx.append(ResultsDataSet(test, [], []))

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifierPytorch = pipeline(feature, model=model, tokenizer=tokenizer, device=0)

    for batch in test_pytorch:
        for i in range(number_of_tests):
            print("Batch_size: ", batch.size)
            dataset = TestDataSet()
            j = 0
            start = time.time()
            for res in classifierPytorch(
                dataset, truncation=True, batch_size=batch.size
            ):
                j = j + 1
                print(j)
            batch.test_time.append(time.time() - start)
            torch.cuda.empty_cache()
            batch.test_list.append(i)
            print("Number of Results: ", i)

    model = ORTModelForSequenceClassification.from_pretrained(
        model_name, from_transformers=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifierONNX = pipeline(feature, model=model, tokenizer=tokenizer, device=0)

    for batch in test_onnx:
        for i in range(number_of_tests):
            print("Batch_size: ", batch.size)
            dataset = TestDataSet()
            j = 0
            start = time.time()
            for res in classifierONNX(dataset, truncation=True, batch_size=batch.size):
                j = j + 1
                print(j)
            batch.test_time.append(time.time() - start)
            torch.cuda.empty_cache()
            batch.test_list.append(i)
            print("Number of Results: ", i)

    xpytorch, ypytorch = cal_results(test_pytorch)
    xonnx, yonnx = cal_results(test_onnx)
    plt.plot(xpytorch[3:], ypytorch[3:], label=f"Pytorch")
    plt.plot(xonnx[3:], yonnx[3:], label=f"Onnx")

    test = "seq_8_last_test"
    plt.xlabel("Rozmiar batcha")
    plt.ylabel("Czas [s]")
    plt.legend()
    plt.show()
    plt.savefig(f"{test}_plot.png")

    savedata(f"{test}_data_pytorch", test_pytorch)
    savedata(f"{test}_data_onnx", test_onnx)


run()
