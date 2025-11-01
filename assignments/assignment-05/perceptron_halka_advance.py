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
                print(f"{epochs} epochs completed | average error: {avg_error}")

        end_time = time.time()
        print(f"--- Training Finished in {(end_time - start_time)*1000:.3f} milliseconds ({epochs} epochs)---\n")

def interactive_console(perceptrons, gate_outputs):
    print("=" * 5 + f" Interactive console ({perceptrons['AND'].input_size} inputs) " + "=" * 5)

    while True:    
        gate_name = input('Enter gate name: ').upper()
        
        if gate_name not in gate_outputs.keys():
            print("gate not available!")
            return
    
        test_input = []
        for i in range(perceptrons[gate_name].input_size):
            ith_input = float(input(f'Enter {i+1}th input (0 or 1): '))
            test_input.append(ith_input)
        
        test_output = perceptrons[gate_name].predict(test_input)
        
        if test_output > 0.5:
            test_output = 1
        else:
            test_output = 0
        
        print(f"\nOutput: {test_output}\n")

        toQuit = input(f'Enter q to quit or retrain with new input size or just enter to continue: ').lower()
        if toQuit == 'q':
            print(f"Quiting current interactive console!\n")
            return

def n_length_bins(n):

    nums = list(range(2**n)) # list a first bracket dile auto (0 to 2^n - 1) range er numbers ashbe
    
    # f"{num:0{n}b}" converts num into n length binary representation string
    # here, colon means conversion
    # 0b means binary
    # n in between 0 and b means length
    # example: f"{8:05b}" = "01000", means 5 length binary repr of 8 as string
    bin_strs = [f'{num:0{n}b}' for num in nums]
    
    bins = [[int(bit) for bit in bin_str] for bin_str in bin_strs]
    
    return np.array(bins)

def get_gate_outputs(inputs):
    return {
        "AND": np.array([np.all(each_input).astype(int) for each_input in inputs]),
        "OR": np.array([np.any(each_input).astype(int) for each_input in inputs]),
        "NAND": np.array([np.logical_not(np.all(each_input)).astype(int) for each_input in inputs]),
        "NOR": np.array([np.logical_not(np.any(each_input)).astype(int) for each_input in inputs])
    }

def main():
    while True:
        input_size = int(input('Enter input size: '))
        
        training_inputs = n_length_bins(input_size)
        gate_outputs = get_gate_outputs(training_inputs)
        
        perceptrons = {}
        for gate, target_outputs in gate_outputs.items():
            print(f"--- Training for {gate} gate ---")
            perceptrons[gate] = Perceptron(input_size=input_size, learning_rate=0.7)
            perceptrons[gate].train(training_inputs, target_outputs)
        
            predicted_outputs = perceptrons[gate].predict(training_inputs)
            predicted_outputs = [1 if output > 0.5 else 0 for output in predicted_outputs]
            
            accuracy = accuracy_score(target_outputs, predicted_outputs)
            report = classification_report(target_outputs, predicted_outputs)
            cm = confusion_matrix(target_outputs, predicted_outputs)
            
            print(f"Accuracy:\n{accuracy*100:.2f}%")
            print(f"Classification report:\n{report}")
            print(f"Confusion matrix:\n{cm}\n")
        
        interactive_console(perceptrons, gate_outputs)
        
        toQuit = input('Enter q to to quit or just enter to retrain with new input size: ').lower()
        if toQuit == 'q':
            print(f"Quiting all processes")
            return

if __name__ == "__main__":
    main()