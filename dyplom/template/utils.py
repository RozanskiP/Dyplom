
def average(arr):
    return sum(arr) / len(arr)
    
def cal_results(list_test):
    x = []
    y = []
    for index, batch in enumerate(list_test):
        x.append(batch.size)
        y.append(average(batch.test_time))
    return x, y