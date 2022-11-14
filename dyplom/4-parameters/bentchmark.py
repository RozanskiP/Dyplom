# READ ME
# aby uruchomic ten bentchamrk nalezy zmienic wersje biblioteki transformers na 2.11.0

from transformers import PyTorchBenchmark, PyTorchBenchmarkArguments
from transformers import TensorFlowBenchmark, TensorFlowBenchmarkArguments


def bentchmark_pytorch(model, batchsizes, sequencelengths):
    print("BENTCHMARK - PYTORCH")
    args = PyTorchBenchmarkArguments(
        models=model,
        batch_sizes=batchsizes,
        sequence_lengths=sequencelengths,
    )
    benchmark = PyTorchBenchmark(args)
    return benchmark.run()


def bentchmark_tensorflow(model, batchsizes, sequencelengths):
    print("BENTCHMARK - TENSORFLOW")
    args = TensorFlowBenchmarkArguments(
        models=model,
        batch_sizes=batchsizes,
        sequence_lengths=sequencelengths,
    )
    benchmark = TensorFlowBenchmark(args)
    return benchmark.run()


def main():
    model_name = ["bert-base-uncased"]
    batch_sizes = [1, 2, 4, 8, 16, 32]
    sequence_lengths = [8, 64, 128, 256, 512]
    result_pytorch = bentchmark_pytorch(model_name, batch_sizes, sequence_lengths)
    # result_tensorflow = bentchmark_tensorflow(model_name, batch_sizes, sequence_lengths)

    print("-------------- PYTORCH --------------")
    print(result_pytorch)
    # print("-------------- TENSORFLOW --------------")
    # print(result_tensorflow)


main()
