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

_SIZE = 8

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
        return _SIZE  # int(2**10)

    def __getitem__(self, i):
        body = "Ahem.. I think I'll be the only one who's saying this but yes, I was a lil bored during the film. Not to say that this is a bad movie, in fact it's a very good attempt at portraying the innermost emotions - dilemma, sorrow, love.., esp it's the director's debut (read from somewhere, is it true?). I felt that something's not quite right, maybe it's just me, I'm not drawn to the characters enough to immerse me in their world. This is a simple story, about ordinary people, ordinary lives. Through simple and short dialogs, the director tries to relate a simple guy's life, and how copes with the news of his illness by laughing it away every time. Oh ya his laughter was kinda cute at first but gradually it gets to me, such a deep hearty roar for a gentle man! I must say, I didn't feel the impact that most readers felt, in fact I was more drawn to the trivial scenarios like spitting of watermelon seeds with his sis that clearly shows that they're comfortable with each other, the granny who came back for another shot - this is kinda melancholic, the thoughtful gesture of writing down the procedures for his dad - hmm but this is predictable.. Don't misunderstood that I'm an action-lover, independent films are my cup of tea! Perhaps I just have a really high expectation after watching many deep films that have stronger imagery. Some Asian films worth the watch: <br /><br />Tony Takitani (depicts loneliness) Wayward Cloud (only 1 dialog) My Sassy Girl (I like it!) 4.30 (loneliness as well) 15 (gangsters lives in local setting) Before sunrise and Before sunset (I just have to mention these even though they are not Asian films. Fans will understand!)"
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

    batch_sizes = list(range(1, 8))
    test_pytorch = []
    for test in batch_sizes:
        test_pytorch.append(ResultsDataSet(test, [], []))
    number_of_tests = 1

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
            print("TIME: ", time.time() - start)
            batch.test_time.append(time.time() - start)
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
            print("TIME: ", time.time() - start)
            batch.test_time.append(time.time() - start)
            batch.test_list.append(i)
            print("Number of Results: ", i)

    xpytorch, ypytorch = cal_results(test_pytorch)
    xonnx, yonnx = cal_results(test_onnx)
    plt.plot(xpytorch, ypytorch, label=f"Pytorch")
    plt.plot(xonnx, yonnx, label=f"Onnx")

    test = "results/test"
    plt.xlabel("Rozmiar batcha")
    plt.ylabel("Czas [s]")
    plt.legend()
    plt.show()
    plt.savefig(f"{test}_plot.png")

    savedata(f"{test}_data_pytorch", test_pytorch)
    savedata(f"{test}_data_onnx", test_onnx)


run()
