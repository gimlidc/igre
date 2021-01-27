import os


def parse(filepath):
    """
    Simple method for fully specified path which split the string into three parts:
    folders, filename without suffix and suffix
    e.g. for ./myfolder/myfile.ext method returns ./myfolder/, myfile, ext
    :param filepath: str
        any form of os.path (relative or absolute)
    :return: str, str, str
        folders, filename without suffix, suffix without dot
    """

    base = os.path.basename(filepath)
    suffix = os.path.splitext(filepath)[1][1:]
    path = filepath[:-len(base)]
    return path, base[:-len(suffix)-1], suffix


def change_suffix(filepath, new_suffix):
    """
    Change suffix for fully specified path
    :param filepath: str
        path to a file (relative or absolute)
    :param new_suffix: str
        suffix of the file without dot e.g. png, jpg, mat, ...
    """
    path, name, suffix = parse(filepath)
    return f"{path}{name}.{new_suffix}"
