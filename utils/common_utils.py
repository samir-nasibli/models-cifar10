import torch

def merge_two_dicts(x, y):
    """
    Merges x and y
    """
    z = x.copy()   # start with x's keys and values
    z.update(y)    # modifies z with y's keys and values & returns None
    return z


def print_cuda_device_info():
    """
    Prints some initial cuda device info
    """
    print(torch.__version__)
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
