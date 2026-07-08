import os


def seperate_out_L(file_name: str) -> int:
    no_post = file_name.split('.')[0]
    L = no_post.split('_')[-1]
    return int(L)


def seperate_out_delay_gain(file_name: str) -> float:
    no_post = ".".join(file_name.split('.')[:1+1])
    delay_gain = no_post.split('_')[-1]
    return float(delay_gain)


def seperate_out_seed(file_name: str) -> int:
    no_post = file_name.split('.')[0]
    seed = no_post.split('_')[-1]
    return int(seed)


def get_files_in_dir_wav(directory: str) -> list:
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')]


def get_files_in_dir_mat(directory: str) -> list:
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.mat')]


def sort_file_path_list(file_paths: list) -> list:
    return sorted(file_paths)