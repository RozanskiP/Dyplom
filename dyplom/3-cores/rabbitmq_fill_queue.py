import pika
import sys
import random
import csv

NUMER_OF_MESSAGE_IN_QUEUE_ = 1000
QUEUE_NAME_ = 'textClass'

rabbitmqhost = "rabbitmq"
rabbitmqport = "5672"

def send_message(channel, routing_key, message):
    channel.basic_publish(exchange='', routing_key=routing_key, body=message)
    print("[x] Sent: " + message)

def fill_in_queue(channel, queue_name, dataset):
    size_of_dataset = len(dataset)
    random_sentence = random.randint(0, size_of_dataset - 1)
    sentence = dataset[random_sentence][0]
    send_message(channel, queue_name, sentence)

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
        queue_size = queue_len(channel, QUEUE_NAME_)
        if queue_size < NUMER_OF_MESSAGE_IN_QUEUE_ - 1:
            fill_in_queue(channel, QUEUE_NAME_, dataset)
    connection.close()

main_func()
