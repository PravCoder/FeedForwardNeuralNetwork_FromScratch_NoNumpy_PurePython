import numpy as np


class RNN:

    def __init__(self, num_layers, hidden_size, vocabulary, inp_seq_length):
        self.num_layers = num_layers
        self.hidden_size = hidden_size          # number of notes for layers, making all layers same size for this implementation
        # each element is an example-input which each represents a input-sequence, each input-sequence is a list of tokens which are one-hot-encoded representing each token-char in that inp-seqeunce
        # but this is stacked so each column represents a one-hot-encoded vector which is the token in that time-step
        self.X = []
        # 1D-array of integers where each number is the index of a token in vocab representing the output-target-character of an input-sequence-example in X
        self.Y = []
        self.vocabulary = vocabulary                 # set of all unique tokens
        self.inp_seq_length = inp_seq_length         # length of the input sequence which is the input of the example, the output is the next character which is the target character the output of this exampel

        # stores hidden states for every layer and for every layer every timestep
        # each element is a key-value-pair where the key is the layer-indx, the value is a dictionary where the keys are the timesteps-int and the value is the hidden-state at layer-l & timestep-t
        # self.h[layer][timestep] is array representing the hidden state vector at that timestep in that layer, hidden state is the vector that stores output of all hidden units at timestep-t
        self.h = {}

        # stores w_ax, w_aa, w_ay, b_a, b_y (only out) parameters for all layers, key is string for that parameter liek "Waa_3" whose value is the matrix for that param at that layer
        self.parameters = {}

    # given a token an char or word it returns a one-hot-encoding of that token based on the token to its vocab index and the size of the vocab, NOT USED
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
    
    def initialize_hidden_states_and_parameters(self):
        # to init hidden state iterate all layers, add a dict for each layer which will contain all timesteps hidden states for that layer
        for l in range(self.num_layers):
            self.h[l] = {}
            self.h[l][0] = np.zeros((self.hidden_size, 1))  # shape of hidden-state at timestep-t for layer-l is (hidden_units, batch_size), for now just init the 0th-ts (do all timesteps in forward pass dynamically)

        # iterate all layers
        for layer in range(self.num_layers):
            # if its the first layer then the input-size of the layer is the size of dict-vocab n_x, else if its not the first layer then the input of that layer is just the hidden-size of that layer which is same for all layers for now
            if layer == 0:
                input_size = len(self.vocabulary)     # input-size is just the number of hidden units in previous layer l-1, if its the first layer then its the vocab-size   
            else:
                input_size = self.hidden_size

            # weights from previous layer l-1 input to cur-hidden-layer connection, shape (n^l_a, n^l-1_a), n^l-1_a = number of hidden units in previous layer l-1
            self.parameters[f"Wax_{layer}"] = np.random.randn(self.hidden_size, input_size) * 0.01
            # weights from connecting hidden state of cur-layer to itself recurring, shape (n^l_a, n^l_a), n^l_a = number of hidden units in current layer l
            self.parameters[f"Waa_{layer}"] = np.random.randn(self.hidden_size, self.hidden_size) * 0.01
            # bias vector of current layer l, vector is of size number of hidden units in cur-layer (n^l_a, 1), bias added to each hidden unit in layer l
            self.parameters[f"ba_{layer}"] = np.zeros((self.hidden_size, 1))

        # weights connecting hidden state to output only a final layer, shape (n_y, n^L_a), n_y = numbe rof possible output classes = len(self.vocabulary)
        self.parameters["Way"] = np.random.randn(len(self.vocabulary), self.hidden_size) * 0.01
        # bias vector for output layer is of size number of output classes, shape (n_y, 1)
        self.parameters["by"] = np.zeros((len(self.vocabulary), 1))

    # given the index of a token of where it lives in vocab convert that into its one-hot-encoding-vector
    def one_hot_encode_token_index(self, token_indx, vocab_size):
        vector = np.zeros((vocab_size, 1))      # create vector of zeros of vocab size
        vector[token_indx] = 1                  # set the index that that token is at in vocab equal to 1 in the vector
        return vector
    
    def vectorize_training_data(self, X, Y):
        # its a 3D-array where each element represents a one-hot-encoded-input-sequence, each of these sequences consists of one-hot-vectors of each token-char in that input-sequence
        X_encoded = []              # (vocab-size, inp-sequence-length, num-examples)

        # an element in X_encoded is a input-sequnce, it is of shape (vocab_size, input_seq_length) for example (26, 4), 
        # where each row represents a character in the vocabulary
        # whwere each col represents a timestep in the input-sequence, we set the inp-seq-len to 4.

        # iterate every input_sequnce in training-inputs
        for input_sequence in X:
            # stores all characters in this input-sequence but one-hot-encoded, so each elemnt is a on-ehot-encoding-list of that character in sequence
            one_hot_sequence = []
            # iterate every token-char in cur-input-sequence
            for token_indx in input_sequence:
                # convert current token-char-index of cur-input-sequence into its one-hot-encoding and add it to this input-sequence list
                one_hot_sequence.append(self.one_hot_encode_token_index(token_indx, len(self.vocabulary)))
                # this stacks each token-char in the shape of (vocab_size, input_sequence_length), where each column is one-hot-vector for that character in the input-sequnce

            # after encoding all token-chars in this seuqnce add the list of all these encoding as a encoded input-sequence-example to X
            X_encoded.append(np.hstack(one_hot_sequence))


        # just indicies not one-hot encoding
        Y_encoded = np.array(Y).flatten()

        self.X, self.Y = X_encoded, Y_encoded
        return X_encoded, Y_encoded
        

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
    rnn = RNN(num_layers=2, hidden_size=3, vocabulary=vocab, inp_seq_length=inp_seq_length)

    print("\nVocab Size: ")
    print(len(rnn.vocabulary))
    print(f"{rnn.num_layers=}")
    print(f"{rnn.hidden_size=}")
    
    # SHOW ONE ENCODING OF TOKEN
    oh_encoding_vector = rnn.one_hot_encode("a", token_to_indx, vocab_size)
    print(f"\nOne-hot encoding of a token: {oh_encoding_vector[0:5]}")

    # CREATE TRAINING EXAMPLES
    print("\nCREATE TRAINING EXAMPLES")
    X, Y = rnn.create_training_example(names=dinosaur_names, token_to_indx=token_to_indx)
    print(X[0:5])
    print(Y[0:5])

    # VECTORIZE TRAINING EXAMPLES
    print("\nVECTORIZE TRAINING EXAMPLES")
    X_encoded, Y_encoded = rnn.vectorize_training_data(X, Y)
    print("X-Training-Example (single example here, the columns are each one-hot character of this single input seqeunce): ")
    print(X_encoded[0])       
    # where each row represents a character in the vocabulary
    # whwere each col represents a timestep in the input-sequence, we set the inp-seq-len to 4.
    print(f"shape: {X_encoded[0].shape} single example shape columns = timesteps, rows = vocab_size.")  
     
    print("\nY-Training-Example (all examples here): ")
    print(Y_encoded)
    print(f"shape: {Y_encoded.shape} entire Y shape")  



    # INITLIAZE HIDDEN STATES
    print("INITIALIZE HIDDEN STATES")
    rnn.initialize_hidden_states_and_parameters()
    print(f"{rnn.h=}")
    """
    self.h = {
        0: {     # Layer 0
            0: h_0_0,   # hidden state at t=0
            1: h_0_1,   # hidden state at t=1
            ...
            T: h_0_T    # hidden state at t=T
        },
        1: {     # Layer 1
            0: h_1_0,
            1: h_1_1,
            ...
            T: h_1_T
        },
        ...
    }
    """


    # INITITALIZE PARAMETERS
    print("\nINITIALIZE PARAMETERS")
    print(f"{rnn.num_layers=}, {rnn.hidden_size=}")
    for param_name, param_value in rnn.parameters.items():
        print(f"{param_name}: {param_value.shape}")



"""
THIS IS FOR THE CREATE TRAINING EXAMPLES
Example 1:
Input: [19, 24, 17, 0] → ['t', 'y', 'r', 'a']
Target: [13] → 'n'

Example 2:
Input: [24, 17, 0, 13] → ['y', 'r', 'a', 'n']
Target: [13] → 'n'

Example 3:
Input: [17, 0, 13, 13] → ['r', 'a', 'n', 'n']
Target: [14] → 'o'

Example 4:
Input: [0, 13, 13, 14] → ['a', 'n', 'n', 'o']
Target: [18] → 's'

Example 5:
Input: [13, 13, 14, 18] → ['n', 'n', 'o', 's']
Target: [0] → 'a'
"""