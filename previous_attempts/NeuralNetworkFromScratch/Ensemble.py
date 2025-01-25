from NNFS import *



"""
The final prediction of ensemble models obtained by combining results from several base models. 
Only for neural network models. Other algorithms coming soon. 
"""
class EnsembleModel:

    def __init__(self, models, problem_type="regression"):
        self.models = models
        self.problem_type = problem_type


    def max_vote(self):
        pass

    def averaging(self, X_example):  # given singular example [value1, value2, value3]
        print(X_example)
        all_models_outputs = []  # [[model1-1, model1-2], [model2-1, model2-2], [model3-1, model3-2]]
        
        for i, model in enumerate(self.models):     # iterate every model in ensemble
            y_pred = model.predict(X_example)       # pass example into current-model and get example
            print(f"Model{i+1} prediction: {y_pred}")
            all_models_outputs.append(y_pred)       # add current-model output to outputs. 

        all_models_outputs = np.mean(all_models_outputs, axis=0) # average all model outputs
        return all_models_outputs


if __name__ == "__main__":
    def generate_noisy_sine_data(num_samples):
        X = np.linspace(0, 2*np.pi, num_samples)
        Y = np.sin(X) + 0.1 * np.random.randn(num_samples)
        return X, Y
    num_samples = 500
    X_train, Y_train = generate_noisy_sine_data(num_samples)
    X_train = X_train.reshape(-1, 1)

    model_1 = NeuralNetwork()
    model_1.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model_1.add(Layer(num_nodes=1, activation=Linear(), initializer=Initializers.glorot_uniform))
    model_1.setup(cost_func=Loss.MSE, input_size=1, optimizer=Optimizers.SGD(learning_rate=0.01))

    model_2 = NeuralNetwork()
    model_2.add(Layer(num_nodes=11, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model_2.add(Layer(num_nodes=1, activation=Linear(), initializer=Initializers.glorot_uniform))
    model_2.setup(cost_func=Loss.MSE, input_size=1, optimizer=Optimizers.SGD(learning_rate=0.01))

    model_3 = NeuralNetwork()
    model_3.add(Layer(num_nodes=12, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model_3.add(Layer(num_nodes=1, activation=Linear(), initializer=Initializers.glorot_uniform))
    model_3.setup(cost_func=Loss.MSE, input_size=1, optimizer=Optimizers.SGD(learning_rate=0.01))


    model_1.train(X_train, Y_train, epochs=100, learning_rate=0.01, batch_size=num_samples, print_cost=False)
    model_2.train(X_train, Y_train, epochs=100, learning_rate=0.01, batch_size=num_samples, print_cost=False)
    model_3.train(X_train, Y_train, epochs=100, learning_rate=0.01, batch_size=num_samples, print_cost=False)

    models_arr = [model_1, model_2, model_3]
    ensemble = EnsembleModel(models=models_arr, problem_type="regression")

    ensemble_pred = ensemble.averaging(X_train[0])
    print(f"Ensemble prediction: {ensemble_pred}")


