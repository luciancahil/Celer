package main.CelerNetwork;

import java.util.Objects;
import java.util.Random;
import java.util.HashSet;

//TODO implement a "run" of the neural network based on data
//TODO implement a cost function
//TODO change number of neurons into an array for each layer

/**
 * The Notation used in this documentation obeys the following conventions:
 * Neuron: N(a, b) - Refers to the bth neuron in the ath layer
 * Bias: B(a, b) -  Refers to the bias of the  bth neuron in the ath layer
 * Weight: W(a, b, c) - Refers to the weight that connects the bth neuron
 *  in the (a - 1)th layer to the cth neuron in the ath layer.
 * Weighted Sum: Z(a, b): Refers to the weighted sum of the bth neuron in the
 *  ath layer
 *
 * Examples:
 * Neuron: N(2, 3) - Refers to the third neuron in the second layer.
 * Bias: B(2, 3) -  Refers to the bias of the third neuron in the second layer
 * Weight: W(3, 1, 2) - Refers to the weight that connects the first neuron
 *  in the second layer to the second neuron in the third layer.
 * Weighted Sum: Z(2, 3): Refers to the weighted sum of the third neuron in the second layer
 *
 * Implementation of a Convolution Neural Network. This neural network uses
 * gradient descent and a minimizing cost function in order to achieve basic
 * machine learning tasks, such as being able to recognize the
 *
 *
 * By default, it will always have 4 layers: an input layer, and output layer,
 * and 2 hidden layers. The input layer feeds into the first hidden
 * layer, which feeds into the second hidden layer, which feeds
 * into the output layer. Thee difference in the number of neurons in
 * first hidden layer and the input layer should by default be 1/3
 * the difference between the number of neurons in the input and output
 * layer, as should the difference in neurons between the fist hidden
 * layer and the second hidden layer, and the second hidden layer and
 * the output layer. Meanwhile, the number of neurons in the input and output
 * layers are supplied by the user.
 *
 * The value of each neuron is called an "activation". It can be any real
 * number for the hidden layers, but must be between 0 and 1 inclusive for the
 * input and output layers.
 *
 * The activation of the input layer is supplied by the user, while the activation
 * of each layer is calculated based on the activation of every neuron in the
 * previous layer.
 *
 * Every neuron in a given layer except the first is connected to each neuron in the previous
 * layer by a weight. The activation of each neuron is multiplied by the corresponding weight,
 * and then a bias is added to each neuron in order to create a weighted sum. That weighted
 * sum is then run through a RELU function for each hidden layer, or through the Sigmoid
 * function for the output layer in order to get the activation of the given neuron.
 *
 * That is, the activation of a given neuron not on the input layer is equal to either the
 * RELU or the sigmoid of the sum of every activation value in previous layer multiplied
 * by a given weight connecting the two neurons plus a given bias.
 */

public class NeuralNetwork {
    // the seed used to generate the initial values in the network
    private final long seed;

    // the array that stores the number of neurons in each layer. numNeuronsLayer[0] stores the
    // number of neurons in the first layer
    private final int[] numNeuronsLayer;

    // the total number of neurons in the entire network
    private final int numNeurons;

    // the number of biases
    private final int numBiases;

    // the number of weights
    private final int numWeights;

    // the number of examples supplied to the network
    private int numExamples;

    // the number of examples we are using to train the network. About 90% of total examples
    private int numTrainingExamples;

    // the number of examples we are using to test the network. About 10% of total examples
    private int numTestingExamples;

    /*
     * The array that stores information about the activation of a given neuron
     * All the neurons in the input layer are stored first, then the first hidden
     * layer, then the second hidden layer, then the output layer.
     *
     * In the input and activation layer, the actual activation is stored.
     *
     * In the hidden layers, the weighted sum is stored, as we need to easily access
     * that value during the gradient descent process.
     *
     */
    private final double[] neurons;

    /*
     * The double array where values of each weight is stored
     * The first numNeuronsL2 * numNeuronsL1 are the neurons between layer 1 and layer 2
     * The next The first numNeuronsL2 * numNeuronsL3 connect layer 2 to layer 3
     * The last The first numNeuronsL3 * numNeuronsL4 connect layer 3 to layer 4
     * The first numNeuronsL2 connect from the first neuron in the first layer
     * to neurons in the second layer, and we always store every connection for the left
     * neuron before moving to the next.
     */
    private final double[] weights;

    /*
     * the double array where the value of each bias is stored
     * the index of each bias corresponds to the index of the neuron in the neurons array - numNeuronsL1.
     * since neurons in the first layer do not have bias
     */
    private final double[] biases;


    /*
     * the double array meant to store the input data that will be used to train the neural network
     */
    private double[][] trainingDataInput;

    /*
     * the double array meant to store the desired output data that will be used to train the neural network
     */
    private double[][] trainingDataOutput;

    /*
     * The double array meant to store the input data that will be used to train the neural network
     * When the training data is provided, 10% will be randomly selected to be used solely
     * for testing purposes
     */
    private double[][] testingDataInput;

    /*
     * the double array meant to store the desired output data that will be used to train the neural network
     * When the training data is provided, 10% will be randomly selected to be used solely
     * for testing purposes
     */
    private double[][] testingDataOutput;


    /**
     * Function: Construction of a brand new neural Network with a randomly generated seed
     * Parameter: inputSize - the number of neurons in the input layer
     * Parameter: outputSize - the number of neurons in the output layer.
     */
    public NeuralNetwork(int inputSize, int outputSize){
        this(inputSize, outputSize, null);
    }

    /**
     * Function: Construction of a brand new neural Network with a previously set seed
     * Parameter: inputSize - the number of neurons in the input layer
     * Parameter: outputSize - the number of neurons in the output layer.
     * Parameter: seed - the seed used to generate the initial values
     */
    public NeuralNetwork(int inputSize, int outputSize, Long seed){
        /* the difference between the number of neurons in the input and output layer */
        int sizeDiff = Math.abs(inputSize - outputSize);

        /* sett the seed value */
        this.seed = Objects.requireNonNullElseGet(seed, () -> (long) (Math.random() * Long.MAX_VALUE));


        /* Setting the number of neurons in each layer */
        numNeuronsLayer = new int[4];
        numNeuronsLayer[0] = inputSize;
        numNeuronsLayer[3] = outputSize;


        /* The difference in size between numNeuronsL1 and numNeuronsL2,
         *  numNeuronsL2 and numNeuronsL3, and between numNeuronsL3 and
         *  numNeuronsL4 should be equal +- 1. That is, the difference
         *  between numNeuronsL1 and numNeuronsL2 is 1/3 the difference
         *  between numNeuronsL1 and numNeuronsL4.
         */

        if(inputSize < outputSize){
            /* there are more neurons in the output layer than input layer*/
            numNeuronsLayer[1] = inputSize + sizeDiff/3;
            numNeuronsLayer[2] = outputSize - sizeDiff/3;
        }else{
            /* there are more neurons in the input layer than output layer*/
            numNeuronsLayer[1] = inputSize - sizeDiff/3;
            numNeuronsLayer[2] = outputSize + sizeDiff/3;
        }

        this.numNeurons = numNeuronsLayer[0] + numNeuronsLayer[1] + numNeuronsLayer[2] + numNeuronsLayer[3];
        this.neurons = new double[numNeurons];

        /* there is a bias for every neuron not in the first layer */
        this.numBiases = numNeurons - numNeuronsLayer[0];
        biases = new double[numBiases];

        this.numWeights = numNeuronsLayer[0]  * numNeuronsLayer[1]  + numNeuronsLayer[1]  * numNeuronsLayer[2]  + numNeuronsLayer[2]  * numNeuronsLayer[3];
        weights = new double[numWeights];

        generateWeights();
    }

    /**
     * Purpose: Generates a random value between -10 and 10 for every weight and bias.
     */
    private void generateWeights(){
        Random rand = new Random(seed);

        for(int i = 0; i < numWeights; i++){
            weights[i] = rand.nextDouble() * 20 - 10;
        }

        for(int i = 0; i < numBiases; i++){
            biases[i] = rand.nextDouble() * 20 - 10;
        }
    }


    /**
     * Purpose: Supplies a large quantity of data to be used for training and testing purposes
     *
     * Parameter: input - the data we will use to feed into the input layer of the neural network. input[i]'s desired output is output[i]
     * Parameter: output - the array of desired outputs from the output layer. input[i]'s desired output is output[i]
     */
    public void setData(double[][] input, double[][] output){
        // Random object used to select which examples are set to training.
        Random rand = new Random(seed);
        HashSet<Integer> isInTesting = new HashSet<Integer>();

        // setting the global variables
        numExamples = input.length;
        numTestingExamples = numExamples / 10;
        numTrainingExamples = numExamples - numTestingExamples;


        // initializing the arrays that will hold the data
        trainingDataInput = new double[numTrainingExamples][numNeuronsLayer[0]];
        trainingDataOutput = new double[numTrainingExamples][numNeuronsLayer[3]];
        testingDataInput = new double[numTrainingExamples][numNeuronsLayer[0]];
        testingDataOutput = new double[numTrainingExamples][numNeuronsLayer[3]];

        // randomly assign values to the testing array

        for(int i = 0; i < numTestingExamples; i++){
            int chosen = rand.nextInt() % numExamples;      // the index of an array chosen to be in the training set

            // we keep generating until we come up with an index not already in the array.
            while(!isInTesting.add(chosen)){
                chosen = rand.nextInt() % numExamples;
            }

            testingDataInput[chosen] = input[chosen];
            testingDataInput[chosen] = output[chosen];
        }


        // assign all values not placed into the testing array into the training array;

        for(int i = 0; i < numExamples; i++){
            if(isInTesting.contains((i))){
                // the value is in the testing array
                i++;
            }else{
                // the value is not in the testing array, so we can put it in the
                trainingDataInput[i] = input[i];
                trainingDataOutput[i] = output[i];
            }
        }
    }

    /**
     * Returns the index of a neuron in the neurons array from the neuron's layer and place in layer
     *
     * @param layer the layer the desired neuron is one
     * @param place the place the neuron is in within the given layer
     * @return the index of the placeth neuron in the layerth layer in the neurons array
     * @throws IllegalArgumentException if an invalid neuron is supplied
     */
    private int getNeuronIndex(int layer, int place) throws IllegalArgumentException{
        if(layer > 4 || layer < 1){
            throw new IllegalArgumentException("The layer number must be between 1 and 4.");
        }else if(place > numNeuronsLayer[layer - 1]){
            throw new IllegalArgumentException("The are not " + place + " neurons in layer " + layer + ".");
        }

        int index = 0;      // the index of the given neuron in the neurons array


        // add up the number of neurons before the given layer
        for(int i = 1; i < layer; i++){
            index += numNeuronsLayer[i - 1];
        }

        // add the number of neurons on the given layer before the inputted neuron
        index += place - 1;

        return index;
    }

    /**
     * Returns the index of the bias of a neuron in the bias array from the neuron's layer and place in layer
     *
     * @param layer the layer the desired neuron is one
     * @param place the place the neuron is in within the given layer
     * @return the index of the bias placeth neuron in the layerth layer in the neurons array
     * @throws IllegalArgumentException if an invalid or input neuron is supplied,
     */
    private int getBiasIndex(int layer, int place) throws IllegalArgumentException{
        if(layer > 4 || layer < 1){
            throw new IllegalArgumentException("The layer number must be between 2 and 4.");
        }else if(layer == 1){
            throw new IllegalArgumentException("The first layer does not have any biases.");
        }else if(place > numNeuronsLayer[layer - 1]){
            throw new IllegalArgumentException("The are not " + place + " neurons in layer " + layer + ".");
        }

        int index = 0;      // the index of the given neuron in the neurons array


        // add up the number of neurons before the given layer
        for(int i = 2; i < layer; i++){
            index += numNeuronsLayer[i - 1];
        }

        // add the number of neurons on the given layer before the inputted neuron
        index += place - 1;

        return index;
    }

    /**
     * Public "get" functions"
     */
    public long getSeed()                   { return seed;}
    public int getNumNeuronsL1()            { return numNeuronsLayer[0];}
    public int getNumNeuronsL2()            { return numNeuronsLayer[1];}
    public int getNumNeuronsL3()            { return numNeuronsLayer[2];}
    public int getNumNeuronsL4()            { return numNeuronsLayer[3];}
    public int getNumNeurons()              { return numNeurons;}
    public int getNumBiases()               { return numBiases;}
    public int getNumWeights()              { return numWeights;}
    public double getNeuron(int index)      { return neurons[index];}
    public double getWeight(int index)      { return weights[index];}
    public double getBias(int index)        { return biases[index];}

    public static void main(String[] args){

    }
}
