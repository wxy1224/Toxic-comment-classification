from os import listdir
from os.path import isfile, join, isdir
import os

def is_dir_exist(path):
    return isdir(path)

def list_files_under_folder(mypath):
    return [f for f in listdir(mypath) if isfile(join(mypath, f))]

def create_folder(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise
