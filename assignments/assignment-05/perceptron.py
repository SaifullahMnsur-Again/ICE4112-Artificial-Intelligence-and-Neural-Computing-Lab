import numpy as np
import time

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = 2 * np.random.rand(input_size) - 1
        self.bias = np.random.rand(1)
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, inputs):
        net_input = np.dot(inputs, self.weights) + self.bias
        return self.sigmoid(net_input)

    def train(self, training_inputs, desired_outputs, max_error):
        print("--- Training Started ---")
        start_time = time.time()

        avg_error = 1e10
        epochs = 1
        
        while avg_error > max_error:
            total_error = 0
            for inputs, desired_output in zip(training_inputs, desired_outputs):
                prediction = self.predict(inputs)
                
                error = desired_output - prediction
                total_error += abs(error) 

                adjustment = self.learning_rate * error
                self.weights += adjustment * inputs
                self.bias += adjustment

            avg_error = total_error[0] / len(training_inputs)
            epochs += 1

        end_time = time.time()
        print(f"--- Training Finished in {end_time - start_time:.9f} seconds ({epochs} epochs)---\n")


def train_and_test_perceptron(gate_name, inputs, outputs, max_error = 0.01):
    print(f"========== Training {gate_name.upper()} Gate ==========")
    
    perceptron = Perceptron(input_size=2, learning_rate=0.7)
    perceptron.train(inputs, outputs, max_error)

    print(f"Final Learned Weights: {perceptron.weights}")
    print(f"Final Learned Bias: {perceptron.bias[0]}")
    print("\n--- Testing Trained Perceptron ---")

    for test_input in inputs:
        prediction = perceptron.predict(test_input)
        final_output = 1 if prediction > 0.5 else 0
        print(f"Input: {test_input} -> Raw Prediction: {prediction[0]:.4f} -> Final Gate Output: {final_output}")
    print("=" * 45 + "\n")
    
    return perceptron

def parse_statement(statement, gate_names):
    nums_gates = []
    numTurn = True
    
    for word in statement.split():
        if numTurn and word.isdigit() and int(word) in [0, 1]:
            nums_gates.append(int(word))
            numTurn = False
        elif not numTurn and word.isalpha() and word.upper() in gate_names:
            nums_gates.append(word.upper())
            numTurn = True
        else:
            return None
    
    if len(nums_gates)%2 == 0:
        return None
    
    return nums_gates

def interective_console(gate_perceptrons):
    gate_names = gate_perceptrons.keys()
    
    while True:        
        statement = str(input('Enter space seperated sequential statement (e.g. 1 AND 0 OR 1 NAND, only 0s, 1s and gates:\n'))
        nums_gates = parse_statement(statement, gate_names)
        
        if nums_gates is None:
            print(f"Incorrect statement!")
            continue
        
        output = nums_gates[0]
        i = 2
        while i < len(nums_gates):
            output = gate_perceptrons[nums_gates[i-1]].predict(np.array([output, nums_gates[i]]))
            if output > 0.5:
                output = 1
            else:
                output = 0
            i += 2
        
        print(f"output: {output}")
        
        

if __name__ == "__main__":
    training_inputs = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    gate_data = {
        "AND": np.array([0, 0, 0, 1]),
        "OR": np.array([0, 1, 1, 1]),
        "NAND": np.array([1, 1, 1, 0]),
        "NOR": np.array([1, 0, 0, 0])
    }
    
    gate_perceptrons = {}
    for gate, data in gate_data.items():
        gate_perceptrons[gate] = train_and_test_perceptron(gate, training_inputs, data)
    
    interective_console(gate_perceptrons)
    
