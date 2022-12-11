import pika
import sys
import random
import csv
import time
import json
import time

Size_q = 0.05

NUMER_OF_MESSAGE_IN_QUEUE_ = 1000
QUEUE_NAME_ = 'textClass'

# rabbitmqhost = "rabbitmq"
rabbitmqport = "5672"
rabbitmqhost = "rabbitmq-clusterip-srv"

def send_message(channel, routing_key, message):
    channel.basic_publish(exchange='', routing_key=routing_key, body=message)
    # print("[x] Sent: " + message)

def fill_in_queue(channel, queue_name, dataset):
    size_of_dataset = len(dataset)
    random_sentence = random.randint(0, size_of_dataset - 1)
    sentence = dataset[random_sentence][0]
    message = {
        "name": sentence,
        "time": time.time()
    }
    send_message(channel, queue_name, json.dumps(message))

def queue_len(channel, queue_name):
    queue = channel.queue_declare(queue=queue_name)
    return queue.method.message_count

def main_func():
    connection = pika.BlockingConnection(pika.ConnectionParameters(rabbitmqhost, rabbitmqport))
    channel = connection.channel()
    file = open('DataSetMovieReviews.csv')
    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    dataset = []
    for row in csvreader:
        dataset.append(row)
    while(True):
        time.sleep(Size_q)
        queue_size = queue_len(channel, QUEUE_NAME_)
        if queue_size < NUMER_OF_MESSAGE_IN_QUEUE_ - 1:
            fill_in_queue(channel, QUEUE_NAME_, dataset)
    connection.close()

main_func(2)
