package main.CelerNetwork;

import java.util.Random;

public class NeuralNetwork {
    // the seed used to generate the initial values in the network
    private final long seed;

    // the number of neurons in the inputLayer
    private final int numNeuronsL1;

    // the number of neurons in the first hidden layer
    private final int numNeuronsL2;

    // the number of neurons in the second hidden layer
    private final int numNeuronsL3;

    // the number of neurons in the output layer
    private final int numNeuronsL4;

    // the total number of neurons in the entire network
    private final int numNeurons;

    // the number of biases
    private final int numBiases;

    // the number of weights
    private final int numWeights;

    // the double array where values of each neuron is stored
    private final double[] neurons;

    /*
     * The double array where values of each weight and bias is stored
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
     * since neurons in the first layer do not have biase
     */
    private final double[] biases;





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
        if(seed == null){
            // no seed was passed
            this.seed = (long)(Math.random() * Long.MAX_VALUE);
        }else {
            // a seed was passed
            this.seed = seed;
        }


        /* Setting the number of neurons in each layer */
        this.numNeuronsL1 = inputSize;
        this.numNeuronsL4 = outputSize;

        /* The difference in size between numNeuronsL1 and numNeuronsL2,
         *  numNeuronsL2 and numNeuronsL3, and between numNeuronsL3 and
         *  numNeuronsL4 should be equal +- 1. That is, the difference
         *  between numNeuronsL1 and numNeuronsL2 is 1/3 the difference
         *  between numNeuronsL1 and numNeuronsL4.
         */

        if(inputSize < outputSize){
            /* there are more neurons in the output layer than input layer*/
            this.numNeuronsL2 = inputSize + sizeDiff/3;
            this.numNeuronsL3 = outputSize - sizeDiff/3;
        }else{
            /* there are more neurons in the input layer than output layer*/
            this.numNeuronsL2 = inputSize - sizeDiff/3;
            this.numNeuronsL3 = outputSize + sizeDiff/3;
        }

        this.numNeurons = numNeuronsL1 + numNeuronsL2 + numNeuronsL3 + numNeuronsL4;
        this.neurons = new double[numNeurons];

        /* there is a bias for every neuron not in the first layer */
        this.numBiases = numNeurons - numNeuronsL1;
        biases = new double[numBiases];

        this.numWeights = numNeuronsL1 * numNeuronsL2 + numNeuronsL2 * numNeuronsL3 + numNeuronsL3 * numNeuronsL4;
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
     * Public "get" functions"
     */
    public long getSeed()                   { return seed;}
    public int getNumNeuronsL1()            { return numNeuronsL1;}
    public int getNumNeuronsL2()            { return numNeuronsL2;}
    public int getNumNeuronsL3()            { return numNeuronsL3;}
    public int getNumNeuronsL4()            { return numNeuronsL4;}
    public int getNumNeurons()              { return numNeurons;}
    public int getNumBiases()               { return numBiases;}
    public int getNumWeights()              { return numWeights;}
    public double getNeuron(int index)      { return neurons[index];}
    public double getWeight(int index)      { return weights[index];}
    public double getBias(int index)        { return biases[index];}

    public static void main(String[] args){

    }
}
