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

    def train(self, training_inputs, desired_outputs):
        print("--- Training Started ---")
        start_time = time.time()

        avg_error = 1e10
        epoch = 1
        
        while avg_error > 1e-3:
            total_error = 0
            for inputs, desired_output in zip(training_inputs, desired_outputs):
                prediction = self.predict(inputs)
                
                error = desired_output - prediction
                total_error += abs(error) 

                adjustment = self.learning_rate * error
                self.weights += adjustment * inputs
                self.bias += adjustment

            avg_error = total_error[0] / len(training_inputs)
            epoch += 1

        end_time = time.time()
        print(f"--- Training Finished in {end_time - start_time:.9f} seconds ---\n")


def run_logic_gate_training(gate_name, inputs, outputs):
    print(f"========== Training {gate_name.upper()} Gate ==========")
    
    perceptron = Perceptron(input_size=2, learning_rate=0.7)
    perceptron.train(inputs, outputs)

    print(f"Final Learned Weights: {perceptron.weights}")
    print(f"Final Learned Bias: {perceptron.bias[0]}")
    print("\n--- Testing Trained Perceptron ---")

    for test_input in inputs:
        prediction = perceptron.predict(test_input)
        final_output = 1 if prediction > 0.5 else 0
        print(f"Input: {test_input} -> Raw Prediction: {prediction[0]:.4f} -> Final Gate Output: {final_output}")
    print("=" * 45 + "\n")


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

    run_logic_gate_training('OR', training_inputs, gate_data['OR'])
    
    # for name, data in gate_data.items():
        # run_logic_gate_training(name, training_inputs, data)
