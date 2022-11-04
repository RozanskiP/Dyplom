import os
import matplotlib.pyplot as plt

from utils import cal_results


def create_directory_if_not_exixts(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def draw_plot(
    directory,
    device,
    title,
    pytorch_results=None,
    onnx_results=None,
    tensorflow_results=None,
):
    create_directory_if_not_exixts(directory)

    if pytorch_results != None:
        xpytorch, ypytorch = cal_results(pytorch_results)
        plt.plot(xpytorch, ypytorch, label=f"Pytorch")

    if onnx_results != None:
        xonnx, yonnx = cal_results(onnx_results)
        plt.plot(xonnx, yonnx, label=f"Onnx")

    if tensorflow_results != None:
        xtensorflow, ytensorflow = cal_results(tensorflow_results)
        plt.plot(xtensorflow, ytensorflow, label=f"Tensorflow")

    plt.xlabel("Batch size")
    plt.ylabel("Time [s]")
    # plt.title(title)
    plt.legend()
    plt.show()
    
    save_plot(directory, title)
    print("File saved in: ", directory)


def save_plot(directory, title):
    plt.savefig(f'{directory}{"_".join(title.split())}.png')
