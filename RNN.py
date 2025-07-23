import numpy as np


class RNN:

    def __init__(self):
        pass

    # given a token an char or word it returns a one-hot-encoding of that token based on the token to its vocab index and the size of the vocab
    def one_hot_encode(self, token, token_to_indx, vocab_size):
        vector = np.zeros((vocab_size, 1))
        vector[token_to_indx[token]] = 1       # get the index that the token represents in the vocab and set that index-position in the vector equal to ``
        return vector

    def foward():
        pass

    def backward():
        pass

    def text_generation():
        pass


# goes through entire dataset and gets all unqiue toekns (characters) and creates mappings from char->indx & indx->char, here a char is the same as a token
# the index here is like where the token lives in the vocab, what position it represents
def get_vocabulary_tokens_mappings(text): 
    vocab = sorted(list(set(text)))         # get all unqiue characters in dataset of text, sort them and create a list this is our vocabulary
    vocab_size = len(vocab)                 # get the size of the vocabulary
    char_to_indx = {ch: i for i, ch in enumerate(vocab)}         # create a mapping from each character in vocab to its index in the vocab
    indx_to_char = {i: ch for i, ch in enumerate(vocab)}         # create a mapping from each index in vocab to that index char in vocab
    return char_to_indx, indx_to_char, vocab_size
    

def load_dinosaur_names_dataset():
    file_path = "datasets/dinosaur_names.txt"
    with open(file_path, "r") as file:
        dinosaur_names = file.readlines()
    dinosaur_names = [name.strip().lower() for name in dinosaur_names]  # strip stuff and put in all lowercase
    return dinosaur_names


if __name__ == "__main__":
    rnn = RNN()
    dinosaur_names = load_dinosaur_names_dataset()
    print(f"Dinosaur names: {dinosaur_names[0:10]}")

    # dino-names is a list of examples so combine it into one string, to get unique tokens in this vocab
    # token_to_indx:  maps token to its position in vocab its index
    # indx_to_token:  maps index in vocab to what token corresponds to it
    token_to_indx, indx_to_token, vocab_size = get_vocabulary_tokens_mappings("".join(dinosaur_names)) 

    print("\nChar->indx dictionary: ")
    count = 0
    for key, value in token_to_indx.items():
        if count < 10:
            print(f"{key}, {value}")
            count += 1
        else:
            break
    
    print("\nIndx->char dictionary: ")
    count = 0
    for key, value in indx_to_token.items():
        if count < 10:
            print(f"{key}, {value}")
            count += 1
        else:
            break

    oh_encoding_vector = rnn.one_hot_encode("a", token_to_indx, vocab_size)
    print(f"One-hot encoding of a token: {oh_encoding_vector[0:5]}")