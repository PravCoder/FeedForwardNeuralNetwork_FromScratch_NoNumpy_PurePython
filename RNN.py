import numpy as np


class RNN:

    def __init__(self, vocabulary, inp_seq_length):
        self.vocabulary = vocabulary                 # set of all unique tokens
        self.inp_seq_length = inp_seq_length

    # given a token an char or word it returns a one-hot-encoding of that token based on the token to its vocab index and the size of the vocab
    def one_hot_encode(self, token, token_to_indx, vocab_size):
        vector = np.zeros((vocab_size, 1))
        vector[token_to_indx[token]] = 1       # get the index that the token represents in the vocab and set that index-position in the vector equal to ``
        print(vector.shape)
        return vector
    
    def create_training_example(self, names, token_to_indx):
        X = []  # is an 2d-array, each element is an input of an example, its [1, 2, 3, 4] its the indicies of each token in vocab which is the sequence of input of an example
        Y = []  # is an 2d-array, each element is an label of an example, its [1] its the index of the toekn in vocab which is the next-token in sequence the label of an example

        # iterate all dinosaur names
        for cur_name in names:
            # for every index in current name
            for i in range(len(cur_name)):  
                # if ith-index plus the input-sequence-length which is where the output-label-token doesnt exceed the length current-name, then proceed
                if (i+self.inp_seq_length) < len(cur_name):   # i+self.inp_seq_length is where the target-character is, less-than because we dont want to go to the length of the cur-name just its last index
                    input_sequence = cur_name[i: i+self.inp_seq_length] # the input-sequence is from the ith-cur-index to the ith plus the inp-seq-length, ending bounds is exclusive so it just gets before the target-character
                    output_token = cur_name[i+self.inp_seq_length]      # the output-token is the ith-index plus the input-seq-length which gets the index of the token after the inp-seq

                    X.append([ token_to_indx[c] for c in input_sequence ]) # for every character in inp-seq convert it into its index in vocab and put it in the same list
                    Y.append([ token_to_indx[output_token] ])              # convert the output-character into its index in vocab and put in the same list

        return X, Y     # ith element in X is the input of an example, and the ith elemnet of Y is the ouput-label of that example

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
    return char_to_indx, indx_to_char, vocab_size, vocab
    

def load_dinosaur_names_dataset():
    file_path = "datasets/dinosaur_names.txt"
    with open(file_path, "r") as file:
        dinosaur_names = file.readlines()
    dinosaur_names = [name.strip().lower() for name in dinosaur_names]  # strip stuff and put in all lowercase
    return dinosaur_names


if __name__ == "__main__":
    dinosaur_names = load_dinosaur_names_dataset()
    print(f"Dinosaur names: {dinosaur_names[0:10]}")

    # dino-names is a list of examples so combine it into one string, to get unique tokens in this vocab
    # token_to_indx:  maps token to its position in vocab its index
    # indx_to_token:  maps index in vocab to what token corresponds to it
    token_to_indx, indx_to_token, vocab_size, vocab = get_vocabulary_tokens_mappings("".join(dinosaur_names)) 

    print("\nThis is our vocabulary: ")
    print(vocab)

    # PRINT TOKEN MAPPINGS
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





    inp_seq_length=4
    rnn = RNN(vocabulary=vocab, inp_seq_length=inp_seq_length)
    
    # SHOW ONE ENCODING OF TOKEN
    oh_encoding_vector = rnn.one_hot_encode("a", token_to_indx, vocab_size)
    print(f"\nOne-hot encoding of a token: {oh_encoding_vector[0:5]}")

    # CREATE TRAINING EXAMPLES
    X, Y = rnn.create_training_example(names=dinosaur_names, token_to_indx=token_to_indx)
    print("\nCREATE TRAINING EXAMPLES")
    print(X[0:5])
    print(Y[0:5])