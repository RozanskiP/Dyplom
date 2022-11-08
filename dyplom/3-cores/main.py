import torch

from ml_service import calculate_pytorch, calculate_onnx, calculate_tensorflow
from draw_plot import draw_plot
from save_results import savedata


def main():
    print("---- CORES ----")
    input_device = -1
    device = torch.device("cpu")

    print("Device: ", device)

    directory = f"plots/{device}/"
    number_of_cores = list(range(1, 17))
    number_of_tests = 4

    pytorch_results = None
    pytorch_results = calculate_pytorch(
        number_of_cores, number_of_tests, input_device, True
    )
    savedata(directory, "pytorch-1", pytorch_results)

    onnx_results = None
    onnx_results = calculate_pytorch(
        number_of_cores, number_of_tests, input_device, False
    )
    savedata(directory, "pytorch-2", onnx_results)

    tensorflow_results = None
    # tensorflow_results = calculate_pytorch(
    #     number_of_cores, number_of_tests, input_device, True, False, 600
    # )
    # savedata(directory, "tensorflow", tensorflow_results)

    draw_plot(
        directory,
        device,
        "CPU",
        pytorch_results,
        onnx_results,
        tensorflow_results,
    )


if __name__ == "__main__":
    main()
