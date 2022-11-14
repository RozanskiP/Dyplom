import torch

from ml_service import calculate_pytorch, calculate_onnx, calculate_tensorflow
from draw_plot import draw_plot
from save_results import savedata


def main():
    print("---- BATCH ----")
    input_device = -1
    if torch.cuda.is_available():
        input_device = 0
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    # device = torch.device("cpu")

    print("Device: ", device)

    directory = f"plots/{device}/"

    batch_sizes = [4, 8, 16, 32, 64]
    # batch_sizes = list(range(1, 33))
    number_of_tests = 2

    pytorch_results = None
    pytorch_results = calculate_pytorch(batch_sizes, number_of_tests, input_device)
    savedata(directory, "pytorch", pytorch_results)

    # batch_sizes = list(range(1, 17))
    onnx_results = None
    onnx_results = calculate_onnx(batch_sizes, number_of_tests, input_device)
    savedata(directory, "onnx", onnx_results)

    tensorflow_results = None
    # tensorflow_results = calculate_tensorflow(batch_sizes, number_of_tests, input_device)
    # savedata(directory, "tensorflow", tensorflow_results)

    draw_plot(
        directory,
        device,
        "GPU_1_1",
        pytorch_results,
        onnx_results,
        tensorflow_results,
    )

if __name__ == "__main__":
    main()
