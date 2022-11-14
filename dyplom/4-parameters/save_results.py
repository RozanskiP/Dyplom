import os

from utils import cal_results


def savedata(directory, file_name, data):

    if not os.path.exists(directory):
        os.makedirs(directory)

    datax, datay = cal_results(data)

    file_w = open(f"{directory}{file_name}.txt", "w")

    for value in datax:
        file_w.write(f"{value}, ")
    file_w.write(f"\n")
    for value in datay:
        file_w.write(f"{value}, ")
    file_w.close()
