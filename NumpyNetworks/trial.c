#include <stdlib.h>
#include <stdio.h>
#include <math.h>



double sigmoid(double x) {return 1 / (1 + exp(-x))}
double dSigmoid(double x) {return x * (1 - x)}

double init_weights() {return ((double)rand()) / ((double )) RAND_MAX)} // with random numbers

void shuffle(int *array, size_t n)
{
    if (n > 1)
    {
        size_t i;
        for (i = 0; i < n - 1; i++)
        {
            size_t j = i + rand() / (RAND_MAX / (n - i) + 1);
            int t = array[j];
            array[j] = array[i];
            array[i] = t;
        }
    }
}

#define numInputs 2  // number of input-nodes
#define numHiddenNodes 2
#define numOutputs 1
#define numTrainingSets 4  // number of examples

int main(void) {
    const double lr = 0.1f;

    double hiddenLayer[numHiddenNodes];  // 1D-list length of number of nodes in hidden-layer
    double outputLayer[numOutputs];     // 1D-list length of number of nodes in output-layer

    double hiddenLayerBias[numHiddenNodes];  // same as hidden-layer size
    double outputLayerBias[numOutputs];      // same as output-layer size

    double hiddenWeights[numInputs][numHiddenNodes];  // weights[prev-input][cur-hidden], w[input-node-i][hidden-node-i],weights from input-layer to hidden-layer
    double outputWeights[numHiddenNodes][numOutputs];  // weights[prev-hidden][cur-output], w[hidden-node-i][output-node-i],weights from hidden-layer to output=layer

    // train_x[number-of-examples][number-of-input-nodes]
    double training_inputs[numTrainingSets][numInputs] = {{0.0f, 0.0f},
                                                          {1.0f, 0.0f},
                                                          {0.0f, 1.0f},
                                                          {1.0f, 1.0f},
    }; // x-data 2D-list for each epoch you have each number of inputs per epoch. When you index a epoch you get a list of inputs for that epoch

    // train_y[number-of-examples][number-of-input-nodes]
    double training_outputs[numTrainingSets][numOutputs] = {{0.0f},
                                                          {1.0f},
                                                          {1.0f},
                                                          {0.0f},
    };
    
    // iterate number of inputs: initialize hidden-layer-weights
    for (int i=0; i<numInputs; i++) {
        // iterate number of hidden nodes
        for (int j=0; j<numHiddenNodes; j++) {
            // weights[input-node-i][hidden-node-in-layer], weights[prev-node-i][cur-node-i]
            hiddenWeights[i][j] = init_weights();  // get random weight value
        }
    }

    // iterate number of hidden nodes: initalize output-layer-weights: 
    for (int i=0; i<numHiddenNodes; i++) {
        // iterate number of output-nodes
        for (int j=0; j<numOutputs; j++) {
            // weights[inputs-i][hidden-node-in-layer], weights[prev-node-i][cur-node-i]
            outputWeights[i][j] = init_weights();
        }
    }

    // iterate number of nodes in hidden-layer: initalize hidden-layer bias
    for (int i=0; i< numHiddenNodes; i++) {
        hiddenLayerBias[i] = init_weights();
    }
    // iterate number of nodes in output-layer: initalize output bias
    for (int i=0; i< numOutputs; i++) {
        outputLayerBias[i] = init_weights();
    }


    int trainingSetOrder[] = {0,1,2,3};

    int numberOfEpochs = 10000;

    // training iterate each epoch
    for (int epoch=0; epochs < numberOfEpochs; epoch++) {
        shuffle(trainingSetOrder, numTrainingSets);
        // iterate number of numTrainingSets or examples 4
        for (int x=0; x<numTrainingSets; x++) {

            int i = trainingSetOrder[x];  // index of input

            // forward pass

            // iterate nodes in hidden-layer: compute hidden layer activation 1st-layer
            for (int j=0; j<numHiddenNodes; j++) {
                double activation = hiddenLayerBias[j];  // init activation sum to bias of jth node in hidden layer
                // iterate number of input nodes
                for (int k=0; k<numInputs; k++) {
                    // train_inputs[i-input-example-indx][k-input-node-indx] * hiddenWeights[k-input-node-indx][i-input-example-indx]
                    activation += training_inputs[i][k] * hiddenWeights[k][j];  // sum products of connections, hiddenWeights[prev][cur]
                }
                hiddenLayer[j] = sigmoid(activation);  // hiddenlayer[node-indx] acctivation
            }

            // iterate nodes in output-layer: compute output-layer-2 activations
            for (int j=0; j<numOutputs; j++) {
                double activation = outputLayerBias[j];   // init activation sum to bias of jth node in output-layer
                // iterate number of hidden-nodes 
                for (int k=0; k<numHiddenNodes; k++) {
                    // hiddenLayer[node-k] * outputWeights[hidden-node-k][output-node-j]
                    activation += hiddenLayer[k] * outputWeights[k][j];  
                }
                outputLayer[j] = sigmoid(activation);  
            }

            printf("Input: %g   Output: %g  Predicted: %g \n", training_inputs[i][0], training_inputs[i][0], training_inputs[i][0])

            // BACKPROPAGATION
            double deltaOutput[numOutputs];

            for (int j=0; j < numOutputs; j++) {
                double error = (training_outputs[i][j] - outputLayer[j]);
                deltaOutput[j] =  error * dSigmoid(outputLayer[j])
            }

            // compute chagne in hidden weights
            double deltaHidden[numHiddenNodes];
            for (j=0; j < numHiddenNodes; j++) {
                double error = 0.0f;
                for (int k=0; k <numOutputs; k++) {
                    error += deltaOutput[k] * outputWeights[j][k]
                }
                deltaHidden[j] = error * dSigmoid(hiddenLayer[j])
            }

            // Apply change in output weights
            for (int j=0; j < numOutputs; j++) {
                outputLayerBias[j] += deltaOutput[j] * lr;
                for (int k=0; k < numHiddenNodes; k++) {
                    outputWeights[k][j] += hiddenLayer[k] * deltaOutput[j] * lr;
                }
            }
            // Apply change in output weights
            for (int j=0; j < numHiddenNodes; j++) {
                hiddenLayerBias[j] += deltaHidden[j] * lr;
                for (int k=0; k < numInputs; k++) {
                    hiddenWeights[k][j] += train_inputs[i][k] * deltaHidden[j] * lr;
                }
            }
        }
    }
}


/* 

XOR Function:
input  |   ouput
0 0    |   0
1 0    |   1
0 1    |   1
1 1    |   0
if the 2 inputs are equal outputs 0, if they differ output 1.

Input nodes: 2
Hidden Layers: 1 with 2 nodes
Output nodes: 1




*/ 

