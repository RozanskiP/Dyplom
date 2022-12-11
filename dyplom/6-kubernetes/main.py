import torch

from ml_service import calculate_pytorch, calculate_onnx, calculate_tensorflow
from draw_plot import draw_plot
from save_results import savedata
import pika
from rabbitmq_fill_queue import main_func
from multiprocessing import Process

QUEUE_NAME_ = "textClass"
rabbitmqhost = "rabbitmq-clusterip-srv"
rabbitmqport = "5672"

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

feature = "text-classification"
model_name = "cmarkea/distilcamembert-base-sentiment"


def queue_len(channel, queue_name):
    queue = channel.queue_declare(queue=queue_name)
    return queue.method.message_count


def average(arr):
    return sum(arr) / len(arr)

def run_queue():
    main_func()

def run_ml():
    input_device = 0
    if torch.cuda.is_available():
        input_device = 0
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Device: ", device)

    connection = pika.BlockingConnection(pika.ConnectionParameters(rabbitmqhost, rabbitmqport))
    channel = connection.channel()

    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    classifierPytorch = pipeline(
        feature, model=model, tokenizer=tokenizer, device=input_device
    )

    channel.queue_delete(queue=QUEUE_NAME_)

    cal_time = []
    i = 5
    while(i < 10):
        queue_size = queue_len(channel, QUEUE_NAME_)
        results = None
        if queue_size > 1:
            results = calculate_pytorch(classifierPytorch, channel)
            cal_time.extend(results)
            print("T: ", results)
            i = i + 1
    print("Average: ", average(cal_time))
    connection.close()

def main():
    print("---- BATCH ----")
    device = torch.device("cpu")

    print("Device: ", device)
    
    Process(target=run_queue).start() 
    Process(target=run_ml).start() 

if __name__ == "__main__":
    main()
