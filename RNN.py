import numpy as np


class RNN:

    def __init__(self):
        pass

    def foward():
        pass

    def backward():
        pass

    def text_generation():
        pass



def load_dinosaur_names_dataset():
    file_path = 'datasets/dinosaur_names.txt'
    with open(file_path, 'r') as file:
        dinosaur_names = file.readlines()
    for name in dinosaur_names:
        print(name.strip())


if __name__ == "__main__":
    load_dinosaur_names_dataset()