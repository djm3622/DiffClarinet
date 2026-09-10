import os


def _value_after_label(file_name: str, label: str) -> str:
    stem = os.path.splitext(os.path.basename(file_name))[0]
    fields = stem.split('_')

    label_index = fields.index(label)
    value = fields[label_index + 1]
    return value


def seperate_out_L(file_name: str) -> int:
    return int(_value_after_label(file_name, 'L'))


def seperate_out_delay_gain(file_name: str) -> float:
    value = _value_after_label(file_name, 'delay')
    if value == 'gain':
        value = _value_after_label(file_name, 'gain')
    return float(value)


def seperate_out_a(file_name: str) -> float:
    return float(_value_after_label(file_name, 'a'))


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
