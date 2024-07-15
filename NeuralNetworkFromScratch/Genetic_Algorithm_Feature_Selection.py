from NNFS import *
from sklearn import datasets
import random

class FeatureSubset:

    def __init__(self, genome):  
        self.genome = genome  # genome = [0, 1, 0, 1, 1, 0,...] # where 1 if ith feature is included 0 if ith feature is excluded
        self.fitness = -1
        self.accuracy = -1
        self.num_features = self.genome.count(1)

    def evaluate_fitness(self, X, Y):
        # create model with genomes features train it and get accuracy
        model = NeuralNetwork()
        model.add(Layer(num_nodes=30, activation=ReLU(), initializer=Initializers.glorot_uniform))
        model.add(Layer(num_nodes=20, activation=ReLU(), initializer=Initializers.glorot_uniform))
        model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
        model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
        model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
        model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
        model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
        model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
        model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
        model.add(Layer(num_nodes=10, activation=ReLU(), initializer=Initializers.glorot_uniform))
        model.add(Layer(num_nodes=1, activation=Sigmoid(), initializer=Initializers.glorot_uniform))

        model.setup(cost_func=Loss.BinaryCrossEntropy, input_size=self.num_features, optimizer=Optimizers.SGD(learning_rate=0.01))
        model.train(X, Y, epochs=10, learning_rate=0.75, batch_size=len(X), print_cost=True)

        self.fitness = model.costs[-1]  # set fitness equal to model cost TBD: change to accuracy
        print(self.fitness)



class Population:

    def __init__(self, num_individuals, num_features):
        self.num_individuals = num_individuals
        self.num_features = num_features
        self.chromosomes = []
        self.parents = []
        self.generation_num = 1
        self.tournament_size = 5

    def create_inital_population(self, id_to_label):
        for _ in range(self.num_individuals):
            cur_genome = [0 for _ in range(self.num_features)] # initalize genome to zeros 
            num_selected_features = np.random.randint(1, self.num_features + 1) # randomly determine the number of features to include in the cur-subset
            selected_indices = np.random.choice(self.num_features, num_selected_features, replace=False) # out of all the numbers in the range of 0 to number of features it randomly selects 'num_selected_features' numbers which are the indicies of the features included
            # print(selected_indices, num_selected_features)
            for indx in selected_indices:
                cur_genome[indx] = 1
            cur_subset = FeatureSubset(genome=cur_genome)
            self.chromosomes.append(cur_subset)

    def create_new_generation(self, X, Y):
        self.evaluate_pop_fitness(X, Y)
        self.select_best_subsets_tournament_selection()
        self.chromosomes = []

        for _ in range(self.num_individuals //  2):
            parent1 = random.choice(self.parents)
            parent2 = random.choice(self.parents)

            offspring1, offspring2 = self.single_point_crossover(parent1, parent2)

            self.chromosomes.append(offspring1)
            self.chromosomes.append(offspring2)

    def evaluate_pop_fitness(self, X, Y):
        for a in self.chromosomes:
            a.evaluate_fitness(X, Y)
    
    def single_point_crossover(self, subset1, subset2):
        crossover_point = random.randint(1, len(subset1.genome)-1) # choose random cross-point frmo indx-1 to last index in the parent1s genome
        genome1 = subset1.genome[:crossover_point] + subset2.genome[crossover_point:]
        genome2 = subset1.genome[crossover_point:] + subset2.genome[:crossover_point]
        offspring1 = FeatureSubset(genome=genome1)
        offspring2 = FeatureSubset(genome=genome2)
        return offspring1, offspring2
    
    def select_best_subsets_tournament_selection(self):
        self.parents = []
        for _ in range(self.num_individuals):
            self.parents.append(self.tournament_selection())

    def tournament_selection(self):
        sample = random.sample(self.chromosomes, self.tournament_size)
        sample.sort(key=lambda rocket: rocket.fitness, reverse=True) # from the randomly selected subsets sort them based on fitness and return the highest fitness subset
        return sample[0]                                            # return the highest fitness of the selected sample
    
    def is_done(self):
        for a in self.chromosomes:
            if a.fitness == -1:
                return False
        return False

def main():
    data = datasets.load_breast_cancer()
    feature_names = list(data.feature_names)
    label_names = list(data.target_names)
    id_to_label = dict(enumerate(feature_names))
    num_features = len(feature_names)
    num_labels = len(label_names)
    X = np.array(data.data)     # each row represents example, each element in row is a feature, lenght of row is 30. shape(examples, feautres)
    Y = np.array(data.target)   # 1D-list where each element is either 0/1 for each element. Shape
    print(f"Number of features: {num_features}")
    print(f"Number of labels: {num_labels}, {label_names}")
    print(f"Id to label: {id_to_label}")

    total_generations = 5
    pop = Population(15, num_features)
    pop.create_inital_population(id_to_label)

    for gen in range(total_generations):
        print(f"Generation #{gen}")
        if pop.is_done() == True:
            pop.create_new_generation(X, Y)

if __name__ == "__main__":
    main()