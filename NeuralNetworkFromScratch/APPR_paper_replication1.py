from NNFS import *
import pandas as pd
import csv 

"""
Paper: "Using Data Mining to Predict Secondary School Student Performance"

"""

class Student:

  all_students = [] # stores all student example objects

  def __init__(self, data):
      self.data = data


  @classmethod
  def predict_g3(cls, model, student, feature_names):
      pass

def show_examples():
  for s in Student.all_students:
    temp = ""
    for key, value in s.data.items():
      temp += f'{key}: {value},   '

    print(temp)

def preprocess_data(all_students, feature_names, output_labels): # convertrs list of student objects into numpy train_X/train_Y arrays
    num_examples = len(all_students)
    num_features = len(feature_names)
    num_outputs = len(output_labels)

    X_train = np.zeros((num_examples, num_features))
    Y_train = np.zeros((num_examples, num_outputs))

    for i, student in enumerate(all_students):
        for j, feature_name in enumerate(feature_names):
            # get ith-student-matrix and jth feature of cur-student and set it equal to the feature value
            X_train[i, j] = student.data.get(feature_name, 0)  # Use 0 as default if feature_name is not present

        for k, output_label in enumerate(output_labels):
            # get the ith student-matrix in output and kth output-node and set it equal to the output-label value
            Y_train[i, k] = student.data.get(output_label, 0)  # Use 0 as default if output_label is not present

    return X_train, Y_train


def main():
    label_encoder = {"yes":1, "no":0}  # converts string feature into numerical value
    feature_names = ["studytime","failures","schoolsup","famsup","activities","paid","internet","higher","absences","G1","G2"] # all features currently being used
    output_labels = ["G3"]    

    df = pd.read_csv("NeuralNetworkFromScratch/student-mat.csv", sep=";")

    for index, row in df.iterrows():
        example_data = {"id":index}
        for column_label, value in row.items():
            c = column_label.split(";")
            # if the feature-name is one we are using OR it is in the output-labels
            if column_label in feature_names or column_label in output_labels:
                # if feature-value is string add it to currrent-example and encode into numerical value
                if isinstance(value, str) == True:
                    example_data[column_label] = label_encoder[value]
                # if feature-value is numerical add it to current-example dictionary with feature-name as key and number as value
                else:
                    example_data[column_label] = value

        s1 = Student(example_data)    # create student-obj for each example and pass in attribute values
        Student.all_students.append(s1)

    # print(feature_names)
    # show_examples()
        
    X_train, Y_train = preprocess_data(Student.all_students, feature_names, output_labels)
    model = NeuralNetwork()

    # Regression Model (the G3 value numeric output between 0 and 20)
    model.add(Layer(num_nodes=64, activation=ReLU(), initializer=Initializers.glorot_uniform))
    model.add(Layer(num_nodes=1, activation=Linear(), initializer=Initializers.glorot_uniform))
    model.setup(cost_func=Loss.RMSE, input_size=len(feature_names), optimizer=Optimizers.SGD(learning_rate=0.01))
    model.train(X_train, Y_train, epochs=5000, learning_rate=0.01, batch_size=len(Student.all_students))


    Y_pred = model.predict(X_train)

    print(f'Actual G3-Score: {Y_train[24]}')
    print(f'Prediction G3-Score: {Y_pred[24]}')
    

main()
   



