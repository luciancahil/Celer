# Celer

## About
This is an implementation of a Convolutional Neural Network in Java.


## How to Use

### Setup

The main function to run is "src/main/Main.java". The easiest way to use this project is to edit public static void main(String args[]);

To create a Neural Net object, make a call similar to the one below:

<pre>
                NeuralNetwork network = new NeuralNetwork(inputSize, outputSize);
</pre>

Where inputSize is an int that represents the number of neurons in the input layer, and outputSize is an int that represents the number of neurons in the outputlayer. For example, for the simple MINST hand-drawn digit dataset, inputSize shoudl be 784 (for all 784 pixels), and outputSize should be 10 (for all 10 digits).

To provide it with data, use the following method after declaring a NeuralNetwork object:

<pre>
                network.setData(inputArray,outputArray);
</pre>

Where inputArray and outputArrays are both arrays of arrays of doubles. A given element of inputArray is the activiation of the input layer, while outputArray represents the expected activation of the outputLayer for its corresponding input. All arrays  in inputArray must have exactly as many elements as the provided inputSize parameter, and all arrays in outputArray must have as many elements as the outputSize parameter.

### Training
To train the network, use the method 
<pre>
                network.setData(batchSize, epochs);
</pre>

Where batchSize represents the number of data sets we want to run through in each batch before adjusting the parameters, and epochs represents how many times we should go through the entire data set.


### Testing

After training is done, we must implement a Test class. It is perfectly fine to use the one provided. Simply declare

<pre>
        TestImplementation test = new TestImplementation(testType);
</pre>

Whre testType is either "SELECTION" or "TRUEFALSE".

Selection means only one neuron should be on in the final output layer, and we want to check if we have the correct one on.

TrueFalse means multiple neurons are on in the output layer, and we want to check if the network lit up the correct ones.

Then, pass the declared test class into:
<pre>
      network.runTests(test);
</pre>
