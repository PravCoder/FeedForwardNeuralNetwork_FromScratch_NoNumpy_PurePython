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


# goes through entire dataset and gets all unqiue toekns (characters) and creates mappings from char->indx & indx->char 
def get_vocabulary_tokens_mappings(text): 
    vocab = sorted(list(set(text)))         # get all unqiue characters in dataset of text, sort them and create a list this is our vocabulary
    vocab_size = len(vocab)                 # get the size of the vocabulary
    char_to_ix = {ch: i for i, ch in enumerate(vocab)}  # create a mapping from each character in vocab to its index in the vocab
    ix_to_char = {i: ch for i, ch in enumerate(vocab)}         # create a mapping from each index in vocab to that index char in vocab
    return char_to_ix, ix_to_char, vocab_size
    


def load_dinosaur_names_dataset():
    file_path = "datasets/dinosaur_names.txt"
    with open(file_path, "r") as file:
        dinosaur_names = file.readlines()
    dinosaur_names = [name.strip().lower() for name in dinosaur_names]  # strip stuff and put in all lowercase
    return dinosaur_names


if __name__ == "__main__":
    dinosaur_names = load_dinosaur_names_dataset()
    print(f"Dinosaur names: {dinosaur_names[0:10]}")

    char_to_ix, ix_to_char, vocab_size = get_vocabulary_tokens_mappings(dinosaur_names)

    print("\nChar->indx dictionary: ")
    count = 0
    for key, value in char_to_ix.items():
        if count < 10:
            print(f"{key}, {value}")
            count += 1
        else:
            break