import numpy as np
import time
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.input_size = input_size
        self.weights = 2 * np.random.rand(input_size) - 1
        self.bias = np.random.rand(1)
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, inputs):
        net_input = np.dot(inputs, self.weights) + self.bias
        return self.sigmoid(net_input)

    def train(self, training_inputs, target_outputs, max_error=.001):
        print("--- Training Started ---")
        start_time = time.time()

        avg_error = 1e10
        epochs = 0
        
        while avg_error > max_error:
            total_error = 0
            for inputs, target_output in zip(training_inputs, target_outputs):
                prediction = self.predict(inputs)
                error = target_output - prediction

                adjustment = self.learning_rate * error
                self.weights += adjustment * inputs
                self.bias += adjustment
                
                total_error += abs(error) 

            avg_error = total_error / self.input_size
            epochs += 1
            if epochs%1000 == 0:
                print(f"{epochs} completed | average error: {avg_error}")

        end_time = time.time()
        print(f"--- Training Finished in {(end_time - start_time)*1000:.3f} milliseconds ({epochs} epochs)---\n")

def main():
    
    input_size = 2
    
    # 2^input_size ta input
    training_inputs = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    
    # per gate 2^input_size ta ouput
    gate_outputs = {
        "AND": np.array([0, 0, 0, 1]),
        "OR": np.array([0, 1, 1, 1]),
        "NAND": np.array([1, 1, 1, 0]),
        "NOR": np.array([1, 0, 0, 0])
    }
    
    gate_name = input('Enter gate name: ').upper()
    
    if gate_name not in gate_outputs.keys():
        print("gate not available!")
        return
    
    target_outputs = gate_outputs[gate_name]
    
    perceptron = Perceptron(input_size=input_size, learning_rate=0.7)
    perceptron.train(training_inputs, target_outputs)
    
    predicted_outputs = perceptron.predict(training_inputs)
    predicted_outputs = [1 if output > 0.5 else 0 for output in predicted_outputs]
    
    accuracy = accuracy_score(target_outputs, predicted_outputs)
    report = classification_report(target_outputs, predicted_outputs)
    cm = confusion_matrix(target_outputs, predicted_outputs)
    
    print(f"Accuracy:\n{accuracy*100:.2f}%")
    print(f"Classification report:\n{report}")
    print(f"Confusion matrix:\n{cm}")    
    
    while True:
        test_input = []
        for i in range(input_size):
            ith_input = float(input(f'Enter {i+1}th input (0 or 1): '))
            test_input.append(ith_input)
        
        test_output = perceptron.predict(test_input)
        
        if test_output > 0.5:
            test_output = 1
        else:
            test_output = 0
        
        print(f"Output: {test_output}")

        toQuit = input(f'Enter q to quit, enter to continue: ').lower()
        if toQuit == 'q':
            return

if __name__ == "__main__":
    main()