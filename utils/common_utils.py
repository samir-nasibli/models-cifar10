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
    if torch.cuda.is_available():
        print(f'Torch {torch.__version__}')
        print(f'Current device {torch.cuda.current_device()}')
        print(torch.cuda.device(0))
        print(f'Devices count: {torch.cuda.device_count()}')
        print(f'Device: {torch.cuda.get_device_name(0)}')
    else:
        print("CUDA is unavailable")


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)
