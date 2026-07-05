import os


def seperate_out_L(file_name: str) -> int:
    no_post = file_name.split('.')[0]
    L = no_post.split('_')[-1]
    return int(L)


def seperate_out_delay_gain(file_name: str) -> float:
    no_post = file_name.split('.')[0]
    delay_gain = no_post.split('_')[-1]
    return float(delay_gain)


def seperate_out_seed(file_name: str) -> int:
    no_post = file_name.split('.')[0]
    seed = no_post.split('_')[-1]
    return int(seed)