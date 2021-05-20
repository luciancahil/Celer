package main.CelerNetwork;

import java.util.Objects;
import java.util.Random;
import java.util.HashSet;
import main.CelerNetwork.NeuralMath.NeuralMath;

//TODO start calculating gradient
//TODO add learning rate calculation function

/**
 * The Notation used in this documentation obeys the following conventions:
 * Cost: C - Refers to the output of the cost function
 * Neuron: N(a, b) - Refers to the bth neuron in the ath layer
 * Bias: B(a, b) -  Refers to the bias of the  bth neuron in the ath layer
 * Weight: W(a, b, c) - Refers to the weight that connects the bth neuron
 *  in the (a - 1)th layer to the cth neuron in the ath layer.
 * Weighted Sum: Z(a, b): Refers to the weighted sum of the bth neuron in the
 *  ath layer
 *  dx/dy: the change in x (cost function, activation, etc) due to y (cost, weight, bias, etc)
 *
 * Examples:
 * Neuron: N(2, 3) - Refers to the third neuron in the second layer.
 * Bias: B(2, 3) -  Refers to the bias of the third neuron in the second layer
 * Weight: W(3, 1, 2) - Refers to the weight that connects the first neuron
 *  in the second layer to the second neuron in the third layer.
 * Weighted Sum: Z(2, 3): Refers to the weighted sum of the third neuron in the second layer
 * dC/dN(1,2) Refers to how much chaning the 2nd neuron in the first layer will affect the cost function
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
 * sum is then run through a leaky RELU function for each hidden layer, or through the Sigmoid
 * function for the output layer in order to get the activation of the given neuron.
 *
 * That is, the activation of a given neuron not on the input layer is equal to either the
 * leaky RELU or the sigmoid of the sum of every activation value in previous layer multiplied
 * by a given weight connecting the two neurons plus a given bias.
 */

public class NeuralNetwork {
    // the seed used to generate the initial values in the network
    private final long seed;

    private final static int NUM_LAYERS = 4;

    // the array that stores the number of neurons in each layer. numNeuronsLayer[0] stores the
    // number of neurons in the first layer
    private final int[] numNeuronsLayer;

    // the total number of neurons in the entire network
    private final int numNeurons;

    // the number of biases
    private final int numBiases;

    // the number of weights
    private final int numWeights;

    // the number of weights pointing to layer 2
    private final int numWeightsTwo;

    // the number of weights pointing to layer 3
    private final int numWeightsThree;

    // the number of weights pointing to layer 4
    private final int numWeightsFour;


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
     * Since the weighted sum of a neuron is required to calculate the gradient,
     * and the activation of a neuron is easy to calculate given the weighted sum,
     * the activation is NEVER stored, and this array only stores the weighted sum.
     */
    private final double[] neuronWeightedSums;

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

    // the output we are currently aiming for in a given round of training
    private double[] currentDesiredOutput;


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
        this.neuronWeightedSums = new double[numNeurons];

        /* there is a bias for every neuron not in the first layer */
        this.numBiases = numNeurons - numNeuronsLayer[0];
        biases = new double[numBiases];


        this.numWeightsTwo = numNeuronsLayer[0]  * numNeuronsLayer[1];
        this.numWeightsThree = numNeuronsLayer[1]  * numNeuronsLayer[2];
        this.numWeightsFour = numNeuronsLayer[2]  * numNeuronsLayer[3];

        System.out.println(numWeightsTwo);
        System.out.println(numWeightsThree);
        System.out.println(numWeightsFour);


        this.numWeights = numWeightsTwo  + numWeightsThree  + numWeightsFour;
        weights = new double[numWeights];

        generateWeights();

    }

    /**
     * Sets every weight and bias to a random value between -10 and 10
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

            testingDataInput[i] = input[chosen];
            testingDataOutput[i] = output[chosen];
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
        validateLayer(layer);

        if(place > numNeuronsLayer[layer - 1] || place < 1){
            throw new IllegalArgumentException("The is no neuron number " + place + " in layer " + layer + ".");
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
            throw new IllegalArgumentException("The layer number must be between 1 and 4.");
        }else if(layer == 1){
            throw new IllegalArgumentException("The first layer does not have any biases.");
        }else if(place > numNeuronsLayer[layer - 1] || place < 1){
            throw new IllegalArgumentException("The is no neuron number " + place + " in layer " + layer + ".");
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
     * Gets the index of a desired weight
     * @param neuronOnePlace the place of the first neuron in its layer
     * @param neuronTwoLayer the layer of the second neuron
     * @param neuronTwoPlace the place of the second neuron in its layer
     * @return the index of the weight in the weights array connecting the 2 neurons
     */
    public int getWeightIndex(int neuronOnePlace, int neuronTwoLayer, int neuronTwoPlace){
        int neuronOneLayer = neuronTwoLayer - 1;

        if(neuronTwoLayer > 4 || neuronTwoLayer <= 1){
            throw new IllegalArgumentException("The second layer number must be between 2 and 4.");
        }else if(neuronOnePlace > numNeuronsLayer[neuronOneLayer - 1] || neuronOnePlace < 1){
            throw new IllegalArgumentException("The is no neuron number " + neuronOnePlace + " in layer " + neuronOneLayer + ".");
        }else if(neuronTwoPlace > numNeuronsLayer[neuronTwoLayer - 1] || neuronTwoPlace < 1){
            throw new IllegalArgumentException("The is no neuron number " + neuronTwoPlace + " in layer " + neuronTwoLayer + ".");
        }

        int index = 0;// the index of the given weight in the weights array


        // count past all the weights that start in a layer before the layer of the first neuron
        for(int i = 1; i < neuronOneLayer; i++){
            index += numNeuronsLayer[i - 1] * numNeuronsLayer[i];
        }

        // count past all the weights that start at a neuron before neuronOne but on the same layer
        index += numNeuronsLayer[neuronTwoLayer - 1] * (neuronOnePlace - 1);

        // we now have reached the weights who begin at the starting neuron.

        // add the number of weights that end on neuronTwoLayer before neuronTwo

        index += neuronTwoPlace - 1;

        return index;
    }


    /**
     * Calculates how far our neural network is from performing optimally, based on a "cost" function.
     *
     * The cost function is calculated by taking the sum of square of the difference between each desired value of the output
     *      array and it's actual value.
     *
     * @param input the array that we used as input to run the neural netwrok
     * @param target the desired output we hope that the output layer looks like
     * @return the value of the cost function
     */
    public double getCost(double[] input, double[] target){
        validateInput(input);
        validateOutput(target);

        // the value of the cost function
        double cost = 0;

        // run the neural network with current parameters
        runExample(input);

        // cycle through each neuron in the output layer, and check it against the target
        for(int i = 1; i <= numNeuronsLayer[NUM_LAYERS - 1]; i++){
            // calculate the difference between actual and desired activation.
            double diff = target[i - 1] - getActivation(NUM_LAYERS, i);

            // add the square of each difference to the cost value
            cost += Math.pow(diff, 2);
        }

        return cost;
    }

    /**
     * Changes the value of each weighted sum based on a given index that serves as the input layer.
     * @param input: The array that will supply information to the input layer.
     * @throws IllegalArgumentException if the input array does not have the same number of entries as the input layer
     */
    public void runExample(double[] input) throws IllegalArgumentException{



        validateInput(input);


        // Fills the input layer of the neural network with data from the input array
        for(int i = 0; i < numNeuronsLayer[0]; i++){
            if(input[i] < 0){
                throw new IllegalArgumentException("Negative values are not permitted as inputs");
            }
            neuronWeightedSums[i] = input[i];
        }


        /*
        Sets the value of the weightedSums array for every layer past the first.
        We are not storing the actual activation; only the weighted sum.
         */
        for(int receivingLayer = 2; receivingLayer <= numNeuronsLayer.length; receivingLayer++){    // for every layer from the second to the last
            for(int receivingNeuron = 1; receivingNeuron <= numNeuronsLayer[receivingLayer - 1]; receivingNeuron++){ // for every neuron in the current layer
                double weightedSum = biases[getBiasIndex(receivingLayer, receivingNeuron)]; // the weighted sum we are calculating.

                // for every neuron in the previous layer
                for(int inputNeuron = 1; inputNeuron <= numNeuronsLayer[(receivingLayer - 1) - 1]; inputNeuron++){
                    // add the activation of every neuron in the previous layer times the weight between said neuron and the recieving neuron
                    weightedSum += NeuralMath.leakyRELU(neuronWeightedSums[getNeuronIndex((receivingLayer - 1), inputNeuron)]) * weights[getWeightIndex(inputNeuron, receivingLayer, receivingNeuron)];
                }

                /* store the weighted sum in the neurons array
                 * we do NOT store tha actual activation in the array, as we need the weighted sum
                 * to calculate the gradient
                 */
                neuronWeightedSums[getNeuronIndex(receivingLayer, receivingNeuron)] = weightedSum;
            }
        }
    }


    /**
     * Checks if a given layer exists in the given network
     * @param layer: the layer we are checking the validity of
     */
    private void validateLayer(int layer){
        if(layer <= 0 || layer > NUM_LAYERS){
            throw new IllegalArgumentException("There is no layer " + layer + ".");
        }
    }

    /**
     * Checks if a given input is valid for the neural network
     * @param array: the array we are checking
     */
    private void validateInput(double[] array){
        if(array.length != numNeuronsLayer[0]){
            throw new IllegalArgumentException("The input array should have " + numNeuronsLayer[0] + " entries instead of " + array.length + ".");
        }
    }

    /**
     * Checks if a given output is valid for the neural network
     * @param array: the array we are checking
     */
    private void validateOutput(double[] array){
        if(array.length != numNeuronsLayer[0]){
            throw new IllegalArgumentException("The output array should have " + numNeuronsLayer[0] + " entries instead of " + array.length + ".");
        }
    }

    /**
     * The function that runs to do the training and learning of the network
     */
    public void train(){
        // a running average of how much we should nudge the biases
        final double[] avgBiasNudge = new double[numBiases];

        // a running average of how much we should nudge the weights
        final double[] avgWeightNudge = new double[numWeights];

        // the bias nudges desired by a single set of data
        final double[] biasNudge = new double[numWeights];

        // the weigh nudges desired by a single set of data
        final double[] weightNudge = new double[numWeights];

        // rather than running through all training data every round, run through a batch
        // this controls how many numbers are in each batch
        final int batchSize = 100;

        // the number of batches we can afford to run with the number of training examples
        final int numBatches = numTrainingExamples / batchSize + 1;

        // the size of the current batch.
        // will be equal to batchSize variable except for the last value, which will
        // include any straggles (if there are 299 examples, the last batch will have 199 members)
        int curBatchSize;

        for(int i = 0; i < numBatches; i++){
            // setting cur batch size

            if(i < numBatches - 1){
                // not the final batch
                curBatchSize = batchSize;
            }else{
                // on final batch
                curBatchSize = batchSize + numTrainingExamples % batchSize;
            }


            // the following loop runs through a single batch

            for(int j = 0; j < curBatchSize; j++){
                // runs the network with the proper data as input
                int dataIndex = j + batchSize * i;

                runExample(trainingDataInput[dataIndex]);
                currentDesiredOutput = trainingDataOutput[dataIndex];

                // set proper values for biasNudge
                calculateBiasNudges(biasNudge);

                // set Proper values for weighNudge
                calculateWeightNudges(weightNudge);

                // add values to the running averages
                NeuralMath.updateRollingAvgs(avgBiasNudge,biasNudge,j);
                NeuralMath.updateRollingAvgs(avgWeightNudge,weightNudge,j);
            }

        }
    }

    /**
     * Calculates how much we should nudge each bias according to the current dataset
     * @param biasNudge: the array that stores the desired nudges
     */
    private void calculateBiasNudges(double[] biasNudge) {
        calculateBiasNudgesL2(biasNudge);
        calculateBiasNudgesL3(biasNudge);
        calculateBiasNudgesL4(biasNudge);
    }


    /**
     * Calculates how much we should nudge each bias in layer 2 according to the current dataset
     * @param biasNudge: the array that stores the desired nudges
     */
    private void calculateBiasNudgesL2(double[] biasNudge) {
        for(int i = 0; i < getNumNeuronsL2(); i++){ // there is an L2 bias for every L2 neuron
            int neuronIndex = getNeuronIndex(2,i + 1);
            int biasIndex = getBiasIndex(2,i + 1);

            biasNudge[biasIndex] = getL2BiasNudge(i);
        }
    }

    /**
     * Gets the desired nudge of a bias in the second layer
     * @param place: the placement of the bias within the second layer
     * @return: the nudge we desire
     */
    private double getL2BiasNudge(int place) {
        if(place >= getNumNeuronsL2()){
            throw new IllegalArgumentException("There is no neuron " + (place + 1) + " in layer 2.");
        }

        /*
         * The goal of the nudges is to minimize the cost function.
         *
         * Nudging a bias in layer 2 cannot directly affect the cost function.
         *
         * Nudging a bias in layer 2 can affect the weighted sum of its corresponding neuron.
         *
         * Due to the chain rule, dC/db = dz/db * dC/dz
         *
         * Since we just add the bias to the weighted sum, dz and db have a 1-1 correspondence
         * based on changes to the bias
         *
         * Therefore, dC/db = dC/dz
         */

        return 0;
    }


    /**
     * Calculates how much we should nudge each bias in layer 3 according to the current dataset
     * @param biasNudge: the array that stores the desired nudges
     */
    private void calculateBiasNudgesL3(double[] biasNudge) {
    }

    /**
     * Calculates how much we should nudge each bias in layer 4 according to the current dataset
     * @param biasNudge: the array that stores the desired nudges
     */
    private void calculateBiasNudgesL4(double[] biasNudge) {
        for(int i = 0; i < getNumNeuronsL4(); i++){ // there is an L4 bias for every L4 neuron
            int neuronIndex = getNeuronIndex(4,i + 1);
            int biasIndex = getBiasIndex(4,i + 1);

            biasNudge[biasIndex] = getL4BiasNudge(i);
        }
    }

    private double getL4BiasNudge(int place) {
        /*
         * The goal of the nudges is to minimize the cost function.
         *
         * Nudging a bias in layer 4 cannot directly affect the cost function.
         *
         * Nudging a bias in layer 4 can affect the weighted sum of its corresponding neuron.
         *
         * Due to the chain rule, dC/dB(4,d) = dz/dB(4,d) * dC/dZ(4,d)
         *
         * Since we just add the bias to the weighted sum, B(4,d) and Z(4,d) have a 1-1 correspondence
         * based on changes to the bias
         *
         * Therefore, dC/dB(4,d) = dC/dZ(4,d)
         */

        return weightedNudgeL4(place);
    }




    /**
     * Calcualates how much we should nudge each weight according to the current data set
     * @param weightNudge: the array that stores the desired nudges
     */
    private void calculateWeightNudges(double[] weightNudge) {
        calculateWeightNudgesL2(weightNudge);
        calculateWeightNudgesL3(weightNudge);
        calculateWeightNudgesL4(weightNudge);
    }

    /**
     * Calcualates how much we should nudge each weight pointing to layer 2 according to the current data set
     * @param weightNudge: the array that stores the desired nudges
     */
    private void calculateWeightNudgesL2(double[] weightNudge) {
    }

    /**
     * Calcualates how much we should nudge each weight pointing to layer 3 according to the current data set
     * @param weightNudge: the array that stores the desired nudges
     */
    private void calculateWeightNudgesL3(double[] weightNudge) {
    }

    /**
     * Calcualates how much we should nudge each weight pointing to  layer 4 according to the current data set
     * @param weightNudge: the array that stores the desired nudges
     */
    private void calculateWeightNudgesL4(double[] weightNudge) {
    }


    /**
     * Returns the desired nudge on a given weighted sum in L4
     * @param place the place of the weighted sum in layer 4
     * @return the desired nudge
     */
    private double weightedNudgeL4(int place) {
        // add 1 because we are using a z array in the loop we passed this from
        int neuronIndex = getNeuronIndex(4, place + 1);

        // the activation we wish we had on the current neuron
        double desiredActivation = currentDesiredOutput[place];

        // activation on the current neuron
        double actualActivation = getActivation(4, place + 1);

        // weighted sum of the neuron we are observing
        double wSum = neuronWeightedSums[neuronIndex];

        /*
         * The goal of the nudges is to minimize the cost function.
         *
         * Nudging a weighed sum on Layer 4 can directly affect the cost function
         *
         * Nudging a weighted sum can directly affect the activation of a last layer neuron.
         *
         * Nudging the activation of a last layer neuron can affect the cost function.
         *
         * As such dC/dZ(4,d) = dA(4,d)/dZ(4,d) * dC/dZ(4,d)
         *
         * Since the cost function is calculated as (desiredActivation - actualActivation) ^2,
         * dC/dZ(4,d) = 2 * (desiredActivation - actualActivation)
         *
         * Since we are on the last layer, the weighted sum becomes the activation through the
         * sigmoid function. Thus, dA(4,d)/dZ(4,d) is just sigmoid'(Z(4,d))
         *
         * The final function is sigmoid'(Z(4,d)) * 2 * (desiredActivation - actualActivation)
         */
        return NeuralMath.sigmoidDeriv(wSum) * (2 * (desiredActivation - actualActivation));
    }

    /*
     * Print functions meant to aid in debugging
     */

    /**
     * Prints out every weight that points to a given neuron
     * @param layer: Layer of the target neuron
     * @param place: Place of target neuron in layer
     */
    public void printWeightsTo(int layer, int place){
        validateLayer(layer);

        if (layer == 1) {
            throw new IllegalArgumentException("Neurons on layer 1 has no weights pointing to them.");
        }


        // prints the number of weights connecting to this layer for each neuron in the previous layer
        for(int i = 1; i <= numNeuronsLayer[(layer - 1) - 1]; i++){
            System.out.print(weights[getWeightIndex(i, layer, place)] + " ");
        }
    }

    /**
     * Prints out the weighted sum of every neuron on a given layer
     * @param layer: The layer whoes neurons we wish to print
     */
    public void printWeightedSums(int layer){
        validateLayer(layer);

        for(int i = 1; i <= numNeuronsLayer[layer - 1]; i++){
            System.out.print(neuronWeightedSums[getNeuronIndex(layer, i)] + " ");
        }
    }

    /**
     * Prints out the weighted sum of every neuron on a given layer
     * @param layer: The layer whoes neurons we wish to print
     */
    public void printBiases(int layer){
        validateLayer(layer);

        if (layer == 1) {
            throw new IllegalArgumentException("Neurons on layer 1 has no biases.");
        }

        for(int i = 1; i <= numNeuronsLayer[layer - 1]; i++){
            System.out.print(biases[getBiasIndex(layer, i)] + " ");
        }
    }

    /**
     * Prints all the activations (not weighted sums) of a layer
     * @param layer the layer whose activations we will print
     */
    public void printActivation(int layer){
        validateLayer(layer);


        for(int i = 1; i <= numNeuronsLayer[layer - 1]; i++){
            System.out.print(getActivation(layer, i) + " ");
        }
    }

    /*
     * Public "get" functions"
     */

    /**
     * Gets the activation of a given neuron
     * @param layer layer of neuron who's activation we want
     * @param number place of neuron in layer
     * @return activation of a given neuron
     */
    public double getActivation(int layer, int number){
        double weightedSum = neuronWeightedSums[getNeuronIndex(layer, number)];
        double activation;

        if(layer < NUM_LAYERS){
            activation = NeuralMath.leakyRELU(weightedSum);
        }else{
            activation = NeuralMath.sigmoid(weightedSum);
        }

        return activation;
    }
    

    public long getSeed()                                                                   { return seed;}
    public int getLayerSize(int layer)                                                      { return numNeuronsLayer[layer - 1];}
    public int getNumNeuronsL1()                                                            { return numNeuronsLayer[0];}
    public int getNumNeuronsL2()                                                            { return numNeuronsLayer[1];}
    public int getNumNeuronsL3()                                                            { return numNeuronsLayer[2];}
    public int getNumNeuronsL4()                                                            { return numNeuronsLayer[3];}
    public int getNumNeurons()                                                              { return numNeurons;}
    public int getNumBiases()                                                               { return numBiases;}
    public int getNumWeights()                                                              { return numWeights;}
    public double getWeightedSum(int layer, int place)                                      { return neuronWeightedSums[getNeuronIndex(layer, place)]; }
    public double getBias(int layer, int place)                                             { return biases[getBiasIndex(layer, place)];}
    public double getWeight(int placeOne, int layerTwo, int placeTwo)         { return weights[getWeightIndex(placeOne, layerTwo, placeTwo)];}
}
