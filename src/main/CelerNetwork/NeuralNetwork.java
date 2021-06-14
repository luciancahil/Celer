package main.CelerNetwork;

import java.util.Objects;
import java.util.Random;
import java.util.HashSet;
import main.CelerNetwork.NeuralMath.NeuralMath;
import main.CelerNetwork.NeuralMath.Test;
import org.jetbrains.annotations.NotNull;


//lol. I thought the above would be easy:
//TODO Add an interface to store testing method
//TODO Change how random values in the test are generated
//TODO Maybe lower the standards for a good batch?
//TODO change the documentation in the getIndex functions to use proper notations

/**
 * The Notation used in this documentation obeys the following conventions:
 * Cost: C - Refers to the output of the cost function
 * Neuron: N(a, b) - Refers to the bth neuron in the ath layer
 * Bias: B(a, b) -  Refers to the bias of the  bth neuron in the ath layer
 * Weight: W(a, b, c) - Refers to the weight that connects the ath neuron
 *  in the (b - 1)th layer to the cth neuron in the bth layer.
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
//TODO test bias nudge in L4
public class NeuralNetwork {
    // the seed used to generate the initial values in the network
    private final long seed;

    // number of layers
    private final static int NUM_LAYERS = 4;

    private final static int LAST_LAYER = NUM_LAYERS - 1;

    // the starting value for the activation and weighted sum nudge arrays
    private final static double DEFAULT_NUDGE = -Math.PI;

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

    // an array to store the desired nudges of activations during a training session.
    private final double[] activationNudges;

    // an array to store the desired nudges of activations during a training session.
    private final double[] weightedSumNudges;

    // we will multiply the nudges by this number before adjusting the biases and weights
    private double learningRate = 1;

    // the amount we lower the learning rate by each time we need to
    private final static double LEARNING_REDUCTION_RATE = 10;

    // once the learning rate becomes this value, we will stop the network's training
    private final static double FINAL_LEARNING_RATE = Math.pow(10,-7);


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
        this.activationNudges = new double[numNeurons];
        this.weightedSumNudges = new double[numNeurons];

        /* there is a bias for every neuron not in the first layer */
        this.numBiases = numNeurons - numNeuronsLayer[0];
        biases = new double[numBiases];


        this.numWeightsTwo = numNeuronsLayer[0]  * numNeuronsLayer[1];
        this.numWeightsThree = numNeuronsLayer[1]  * numNeuronsLayer[2];
        this.numWeightsFour = numNeuronsLayer[2]  * numNeuronsLayer[3];


        this.numWeights = numWeightsTwo  + numWeightsThree  + numWeightsFour;
        weights = new double[numWeights];

        generateWeights();

    }

    /**
     * Sets every weight and bias to a random value between -10 and 10
     */
    private void generateWeights(){
        Random rand = new Random(seed);
        int largestLayer;

        if(numNeuronsLayer[0] > numNeuronsLayer[3]){
            // the first layer has as many or more neurons than the last
            largestLayer = numNeuronsLayer[0];
        }else{
            largestLayer = numNeuronsLayer[3];
        }

        for(int i = 0; i < numWeights; i++){
            weights[i] = rand.nextDouble() * 2.0/largestLayer - 1.0/largestLayer;
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
        int skip = 0;

        // setting the global variables
        numExamples = input.length;
        numTestingExamples = numExamples / 10;
        numTrainingExamples = numExamples - numTestingExamples;


        // initializing the arrays that will hold the data
        trainingDataInput = new double[numTrainingExamples][numNeuronsLayer[0]];
        trainingDataOutput = new double[numTrainingExamples][numNeuronsLayer[3]];
        testingDataInput = new double[numTestingExamples][numNeuronsLayer[0]];
        testingDataOutput = new double[numTestingExamples][numNeuronsLayer[3]];

        // randomly assign values to the testing array

        for(int i = 0; i < numTestingExamples; i++){
            int chosen = Math.abs(rand.nextInt()) % numExamples;      // the index of an array chosen to be in the training set

            // we keep generating until we come up with an index not already in the array.
            while(!isInTesting.add(chosen)){
                chosen = Math.abs(rand.nextInt()) % numExamples;
            }
            testingDataInput[i] = input[chosen];
            testingDataOutput[i] = output[chosen];
        }


        // assign all values not placed into the testing array into the training array;

        for(int i = 0; i < numTrainingExamples; i++){
            while(isInTesting.contains((i + skip))){
                // the value is in the testing array
                skip++;
            }

            // the value is not in the testing array, so we can put it in the
            if(i == numTrainingExamples){
                continue;
            }

            trainingDataInput[i] = input[i];
            trainingDataOutput[i] = output[i];

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
        runExample(input, target);

        // cycle through each neuron in the output layer, and check it against the target
        for(int i = 1; i <= numNeuronsLayer[LAST_LAYER]; i++){
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
    public void runExample(double[] input, double[] output) throws IllegalArgumentException{



        validateInput(input);
        validateOutput(output);

        this.currentDesiredOutput = output;


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
        /*
         * We are trying to calculate the desired "nudges" for weights and biases of a given
         * batch in the following loop.
         *
         * A "nudge" is a small change in value.
         *
         * We first divide our total training data into batches of 100 examples.
         *
         * Then, we run through each batch of examples.
         *
         * When running through a single batch, we will calculate how much it would like to change
         * every single weight and bias. We call the desired change a "nudge", and will store
         * them in the biasNudge and weightNudge arrays.
         *
         * A very important piece of information in the  nudge arrays is the ratio between parts.
         * If B(1,2) has a desired nudge of 1 while B(1,3) has a desired nudge of 2, it means that
         * a change to B(1,3) will do twice as much work as a similar nudge to B(1,2). We should
         * change the ones with more effect more, as it is possible a part with a small effect in
         * this sample will have a large effect in another.
         *
         * After we run through the batch, we will add it to the running average arrays, that stores
         * how much and in which direction on average every sample in the batch wants to change
         * every weight and bias.
         *
         * We will then multiply the average desired change by a "learning rate", and then add the
         * desired changes to the array storing the values of the Weights and biases. If we ever
         * reach a point where adding this gives us a higher average cost function, we will divide the
         * learning rate by 10, and try again.
         *
         * Once the learning rate becomes 1E-7, we have completed training.
         *
         */

        // a running average of how much we should nudge the biases
        final double[] avgBiasNudge = new double[numBiases];

        // a running average of how much we should nudge the weights
        final double[] avgWeightNudge = new double[numWeights];

        // the bias nudges desired by a single set of data
        final double[] biasNudge = new double[numBiases];

        // the weigh nudges desired by a single set of data
        final double[] weightNudge = new double[numWeights];

        // rather than running through all training data every round, run through a batch
        // this controls how many numbers are in each batch
        final int batchSize = 100;

        // the number of batches we can afford to run with the number of training examples
        final int numBatches = numTrainingExamples / batchSize;

        // the size of the current batch.
        // will be equal to batchSize variable except for the last value, which will
        // include any straggles (if there are 299 examples, the last batch will have 199 members)
        int curBatchSize;

        // the average cost before we made any nudges
        double averagePreCost;

        // the average cost of a batch after we made nudges
        double averagePostCost;

        // the number of batches in a row that have had their costs increase due to nudges
        int badBatches = 0;

        // the number of bad batchs we need in a row before we reduce the learning rate
        final int BAD_BATCH_TOLERANCE = 5;

        // a batch needs an average cost less than 0.001^2 * number of neurons in the last batch to be considered good
        double goodBatchTolerance = Math.pow(0.001,2) * numNeuronsLayer[LAST_LAYER];

        // number of "good" batches, that meet our desired low cost
        int numGoodBatches = 0;

        // how many cycles we've undergone through very batch
        int rounds = 0;

        //
        int maxRounds = 1000000;


        while(learningRate > FINAL_LEARNING_RATE && rounds < maxRounds && (numGoodBatches < numBatches)) {
            // we will stop running the cycle after we have reached max rounds,
            // the learning rate has reached its minimum value,
            // or each batch is consecutively deemed "good"

            rounds++;
            for (int i = 0; i < numBatches; i++) {
                averagePreCost = 0;


                // setting all values in the Z and A nudge arrays back to the default value
                resetNudgeArrays();

                if (i < numBatches - 1) {
                    // not the final batch
                    curBatchSize = batchSize;
                } else {
                    // on final batch
                    curBatchSize = batchSize + numTrainingExamples % batchSize;
                }



                zeroArrays(avgBiasNudge);
                zeroArrays(avgWeightNudge);

                // the following loop runs through a single batch
                for (int j = 0; j < curBatchSize; j++) {
                    // index of the current data inside the training data array
                    int dataIndex = j + batchSize * i;

                    // the cost function based on this one data set
                    double curCost;

                    // run the network on the data set we want
                    runExample(trainingDataInput[dataIndex], trainingDataOutput[dataIndex]);

                    curCost = getCost(trainingDataInput[dataIndex], trainingDataOutput[dataIndex]);

                    averagePreCost = NeuralMath.updateRollingAvg(averagePreCost, curCost, (j + 1));


                    // set proper values for biasNudge
                    calculateBiasNudges(biasNudge);

                    // set Proper values for weighNudge
                    calculateWeightNudges(weightNudge);

                    // add values to the running averages
                    NeuralMath.updateRollingAvgs(avgBiasNudge, biasNudge, j + 1);
                    NeuralMath.updateRollingAvgs(avgWeightNudge, weightNudge, j + 1);
                }

                // add the nudges to the values of the weights and biases
                for (int k = 0; k < numBiases; k++) {
                    biases[k] = biases[k] + learningRate * avgBiasNudge[k];
                }


                for (int k = 0; k < numWeights; k++) {
                    weights[k] = weights[k] + learningRate * avgWeightNudge[k];
                }


                // get the cost function after nudges
                averagePostCost = 0;
                for (int k = 0; k < batchSize; k++) {
                    int dataIndex = k + batchSize * i;
                    double curCost = getCost(trainingDataInput[dataIndex], trainingDataOutput[dataIndex]);

                    averagePostCost = NeuralMath.updateRollingAvg(averagePostCost, curCost, (k + 1));
                }

                // check if the cost function has been lowered
                if (averagePostCost < averagePreCost) {
                    // the cost function has been lowered
                    badBatches = 0;

                    // check if the current batch is good
                    if(averagePostCost < goodBatchTolerance){
                        // current batch is good. increment the good batches count
                        numGoodBatches++;
                    }else{
                        /// current batch is not good. restart the good batches cycle
                        numGoodBatches = 0;
                    }
                } else {
                    // the cost function has increased. We have a bad batch, and may need to decrease the learning rate

                    // undo the nudges to the values of the weights and biases
                    for (int k = 0; k < numBiases; k++) {
                        biases[k] = biases[k] - learningRate * avgBiasNudge[k];
                    }

                    for (int k = 0; k < numWeights; k++) {
                        weights[k] = weights[k] - learningRate * avgWeightNudge[k];
                    }

                    if (badBatches >= BAD_BATCH_TOLERANCE) {
                        // we have had many bad batches in a row. It is time to reduce the learning rate

                        // reduce the learning rate
                        learningRate /= LEARNING_REDUCTION_RATE;
                        badBatches = 0;
                    } else {
                        badBatches++;
                    }
                }

            }

        }

        System.out.println("Done!");
        System.out.println("Rounds: " + rounds);
        System.out.println("Learning Rate: "+ learningRate);
        System.out.println("Good batche percentage: " + (double)numGoodBatches/numBatches);
    }

    private void zeroArrays(double[] arr){
        for(int i = 0; i < arr.length; i++){
            arr[i] = 0;
        }
    }

    /**
     * Sets every value in the arrays to be the default value
     */
    private void resetNudgeArrays() {
        for(int i = 0; i < numNeurons; i++){
            activationNudges[i] = DEFAULT_NUDGE;
            weightedSumNudges[i] = DEFAULT_NUDGE;
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
     * Calcualates how much we should nudge each weight according to the current data set
     * @param weightNudge: the array that stores the desired nudges
     */
    private void calculateWeightNudges(double[] weightNudge) {
        calculateWeightNudgesL2(weightNudge);
        calculateWeightNudgesL3(weightNudge);
        calculateWeightNudgesL4(weightNudge);
    }

    /**
     * Returns the desired nudge of an activation in the final layer. Will be positive if
     * we want the activation to increase, and negative if we want the activation to increase
     * @param place the place of the neuron in the last layer
     * @return the desired nudge of an activation in the final layer
     */
    private double getActivationNudgeL4(int place){
        // the activation we wish we had on the current neuron
        double desiredActivation = currentDesiredOutput[place - 1];

        // activation on the current neuron
        double actualActivation = getActivation(4, place);

        // index of the neuron in the neuron array
        int index = getNeuronIndex(4, place);

        // value already stored in the activation Nudges array
        double preValue = activationNudges[index];

        if(preValue != DEFAULT_NUDGE){
            // the only way we are NOT at the default alue is we already calculated the necessary nudge.
            // just return that
            return preValue;
        }

        /*
         * Activations of the final layer can directly affect the cost function.
         *
         * The cost function will be minimized by getting the actual activation as
         * close to the desired activation as possible.
         *
         * Since the cost function is (dA - aA)^2, the change in the cost function
         * based on a change in activation is 2 * (da - aA).
         *
         * dA - aA is used over aA - dA so the result will be positive if dA > aA,
         * and we therefore want aA to increase
         */

        activationNudges[index] =2 * (desiredActivation - actualActivation);
        return activationNudges[index];
    }

    /**
     * Returns the desired nudge on a given weighted sum in L4
     * @param place the place of the weighted sum in layer 4
     * @return the desired nudge
     */
    private double getWeightedSumNudgeL4(int place) {
        // add 1 because we are using a zero array in the loop we passed this from
        int neuronIndex = getNeuronIndex(4, place);

        // weighted sum of the neuron we are observing
        double wSum = neuronWeightedSums[neuronIndex];

        // index of the neuron in the neuron array
        int index = getNeuronIndex(4, place);
        double preValue = weightedSumNudges[index];

        if(preValue != DEFAULT_NUDGE){
            // the only way we are NOT at the default alue is we already calculated the necessary nudge.
            // just return that
            return preValue;
        }

        /*
         * Nudging a weighed sum on Layer 4 cannot directly affect the cost function
         *
         * Nudging a weighted sum can directly affect the activation of a last layer neuron.
         *
         * The ratio in the change between the change in a last layer activation and a last
         * layer weighted sum is equal to the derivative of the sigmoid function at the
         * value of the current weighed sum. Therefore, we will multiply the desired
         * activation nudge by that value.
         */

        weightedSumNudges[index] = NeuralMath.sigmoidDeriv(wSum) * getActivationNudgeL4(place);
        return weightedSumNudges[index];
    }

    /**
     * Calculates how much we should nudge each bias in layer 4 according to the current dataset
     * @param biasNudge: the array that stores the desired nudges
     */
    private void calculateBiasNudgesL4(double[] biasNudge) {
        for(int i = 1; i <= getNumNeuronsL4(); i++){ // there is an L4 bias for every L4 neuron
            int biasIndex = getBiasIndex(4,i );

            biasNudge[biasIndex] = getL4BiasNudge(i);
        }
    }


    /**
     * returns the desired nudge of a bias in layer 4
     * @param place the place of the neuron in layer 4
     * @return the desired nudge of a bias in layer 4
     */
    private double getL4BiasNudge(int place) {
        /*
         * Nudging a bias in layer 4 cannot directly affect the cost function.
         *
         * Nudging a bias in layer 4 can affect the weighted sum of its corresponding neuron.
         *
         * Due to the chain rule, dC/dB(4,d) = dz(4,d)/dB(4,d) * dC/dZ(4,d)
         *
         * Since we just add the bias to the weighted sum, B(4,d) and Z(4,d) have a 1-1 correspondence
         * based on changes to the bias. In other words, dz(4,d)/dB(4,d) = 1
         *
         * Therefore, dC/dB(4,d) = dC/dZ(4,d)
         */

        return getWeightedSumNudgeL4(place);
    }


    /**
     * Calcualates how much we should nudge each weight pointing to  layer 4 according to the current data set
     * @param weightNudge: the array that stores the desired nudges
     */
    private void calculateWeightNudgesL4(double[] weightNudge) {
        // one weight connects every weight in Layer 3 to every neuron in layer 4
        int index;

        for(int startPlace = 1; startPlace <= numNeuronsLayer[2]; startPlace++){ // for every neuron in layer 3
            for(int endPlace = 1; endPlace <= numNeuronsLayer[3]; endPlace++){ // for every neuron in layer 4

                index = getWeightIndex(startPlace,4,endPlace);
                weightNudge[index] = getWeightNudgeL4(startPlace,endPlace);
            }
        }
    }

    /**
     * Gets the nudge of a specific weight that points to a neuron in layer 4
     * @param startPlace the place of the starting neuron in layer 3
     * @param endPlace the place of the starting neuron in layer 4
     * @return the desired nudge of the given weight
     */
    private double getWeightNudgeL4(int startPlace, int endPlace) {
        /*
         * Nudging a weight pointing to layer 4 cannot directly affect the cost function.
         *
         * Nudging a weight pointing to layer 4 can affect the weighted sum of its
         * corresponding neuron.
         *
         * Due to the chain rule, dC/dW(c,4,d) = dz(4,d)/dW(c,4,d) * dC/dZ(4,d)
         *
         * The effect of W(c,4,d) on dZ(4,d) is equal to the product of W(c,4,d) and A(3,c).
         * Therefore,the effect of changing W(c,4,d) is exactly proportional to A(3,c).
         *
         * Therefore, dC/dB(4,d) = A(3,c)
         */
        double n1Activation = getActivation(3, startPlace);
        double n2weightedSumNudge = getWeightedSumNudgeL4(endPlace);
        return n1Activation * n2weightedSumNudge;
    }

    /**
     * Returns the desired nudge of the activation of a neuron in Layer 3
     * @param place the place of the neuron in the layer
     * @return the desired nudge of an activation
     */
    private double getActivationNudgeL3(int place){
        // the desired nudges summed up over all neurons in the 4th layer
        double totalNudges = 0;

        // neuron
        int index = getNeuronIndex(3, place);

        // value already stored in the activation Nudges array
        double preValue = activationNudges[index];

        if(preValue != DEFAULT_NUDGE){
            // the only way we are NOT at the default value is we already calculated the necessary nudge.
            // just return that
            return preValue;
        }


        /*
         * Changing the activation of a neuron in layer 3 cannot directly affect the cost function.
         *
         * However, it can directly affect the weighted sum of every neurons in layer 4.
         *
         * Due to the chain rule, dC/dA(3,c) = Σ(dC/dZ(4,d) * dZ(4,d)/dA(3,c))
         *
         * We will need to take into account how nudging the activation in l3 will affect the
         * cost function through EVERY weighted sum in L4. If all of them want A(3,c) to increase,
         * then this value wants to strongly increase. If it's about half, this value will want
         * no particular change. Therefore, we simply add up the desired changes raw.
         *
         * For one instance of dZ(4,d) / dA(3,c), the change is proportionate to the weight
         * connecting neuron (3,c) to neuron (4,d). Therefore, dZ(4,d) / dA(3,c) = W(c,4,d).
         *
         * dC/dA(3,c) = Σ(W(c,4,d) * dC/dZ(4,d))
         */

        for(int i = 1; i <= numNeuronsLayer[3]; i++){
            double curChange = weights[getWeightIndex(place, 4, i )] * getWeightedSumNudgeL4(i);

            totalNudges += curChange;
        }

        return totalNudges;
    }

    /**
     * Returns the desired nudge on a given weighted sum in L3
     * @param place the place of the weighted sum in layer 3
     * @return the desired nudge
     */
    private double getWeightedSumNudgeL3(int place) {
        // add 1 because we are using a zero array in the loop we passed this from
        int neuronIndex = getNeuronIndex(3, place);

        // weighted sum of the neuron we are observing
        double wSum = neuronWeightedSums[neuronIndex];

        // value already calculated
        double preValue = weightedSumNudges[neuronIndex];

        if(preValue != DEFAULT_NUDGE){
            // the only way we are NOT at the default value is we already calculated the necessary nudge.
            // just return that
            return preValue;
        }

        /*
         * Nudging a weighed sum on Layer 3 cannot directly affect the cost function
         *
         * Nudging a weighted sum can directly affect the activation of a layer 3 neuron.
         *
         * The ratio in the change between the change in a layer 3 activation and a layer
         * 3 weighted sum is equal to the derivative of the RELU function at the
         * value of the current weighed sum. Therefore, we will multiply the desired
         * activation nudge by said derivative
         */

        weightedSumNudges[neuronIndex] = NeuralMath.reluDeriv(wSum) * getActivationNudgeL3(place);
        return weightedSumNudges[neuronIndex];
    }


    /**
     * Calculates how much we should nudge each bias in layer 3 according to the current dataset
     * @param biasNudges: the array that stores the desired nudges
     */
    private void calculateBiasNudgesL3(double[] biasNudges) {
        for(int i = 1; i <= numNeuronsLayer[2]; i++){
            int index = getBiasIndex(3, i);

            biasNudges[index] = getBiasNudgeL3(i);
        }
    }

    /**
     * returns the desired nudge of a bias in layer 3
     * @param place the place of the neuron in layer 3
     * @return the desired nudge of a bias in layer 3
     */
    private double getBiasNudgeL3(int place) {
        /*
        It's just the nudge of the weighted sum of the neuron this bias is attached to
         */

        return getWeightedSumNudgeL3(place);
    }

    /**
     * Calculates how much we should nudge each weight pointing to  layer 4 according to the current data set
     * @param weightNudge: the array that stores the desired nudges
     */
    private void calculateWeightNudgesL3(double[] weightNudge) {
        // one weight connects every weight in Layer 3 to every neuron in layer 4
        int index;

        for(int startPlace = 1; startPlace <= numNeuronsLayer[1]; startPlace++){ // for every neuron in layer 2
            for(int endPlace = 1; endPlace <= numNeuronsLayer[2]; endPlace++){ // for every neuron in layer 3
                // add 1, because the for loop starts at 0
                index = getWeightIndex(startPlace,3,endPlace);
                weightNudge[index] = getWeightNudgeL3(startPlace,endPlace);
            }
        }
    }

    /**
     * Gets the nudge of a specific weight that points to a neuron in layer 3
     * @param startPlace the place of the starting neuron in layer 2
     * @param endPlace the place of the starting neuron in layer 3
     * @return the desired nudge of the given weight
     */
    private double getWeightNudgeL3(int startPlace, int endPlace) {
        /*
         * Nudging a weight pointing to layer 3 cannot directly affect the cost function.
         *
         * Nudging a weight pointing to layer 3 can affect the weighted sum of the neuron
         * it is pointing to.
         *
         * Due to the chain rule, dC/dW(b,3,c) = dz(3,c)/dW(b,3,c) * dC/dZ(3,c)
         *
         * The effect of dW(b,3,c) on dZ(3,c) is equal to the product of dW(b,3,c) and A(2,b).
         * Therefore,the effect of changing dW(b,3,c) is exactly proportional to A(2,b).
         *
         * Therefore, dz(3,c)/dW(b,3,c) = A(2,b)
         */
        double n1Activation = getActivation(2, startPlace);
        double n2weightedSumNudge = getWeightedSumNudgeL3(endPlace);
        return n1Activation * n2weightedSumNudge;
    }

    /**
     * Returns the desired nudge of the activation of a neuron in Layer 2
     * @param place the place of the neuron in the layer
     * @return the desired nudge of an activation
     */
    private double getActivationNudgeL2(int place){
        // the desired nudges summed up over all neurons in the 4th layer
        double totalNudges = 0;

        // neuron
        int index = getNeuronIndex(2, place);

        // value already stored in the activation Nudges array
        double preValue = activationNudges[index];

        if(preValue != DEFAULT_NUDGE){
            // the only way we are NOT at the default alue is we already calculated the necessary nudge.
            // just return that
            return preValue;
        }


        /*
         * Changing the activation of a neuron in layer 2 cannot directly affect the cost function.
         *
         * However, it can directly affect the weighted sum of every neurons in layer 4.
         *
         * Due to the chain rule, dC/dA(2,b) = Σ(dC/dZ(3,c) * dZ(3,c)/dA(2,b))
         *
         * We will need to take into account how nudging the activation in l3 will affect the
         * cost function through EVERY weighted sum in L4. If all of them want A(3,c) to increase,
         * then this value wants to strongly increase. If it's about half, this value will want
         * no particular change. Therefore, we simply add up the desired changes raw.
         *
         * For one instance of dZ(3,c) / dA(2,b), the change is proportionate to the weight
         * connecting neuron (2,b) to neuron (3,c). Therefore, dZ(3,c) / dA(2,b) = W(b,3,c).
         *
         * dC/dA(2,b) = Σ(W(b,3,c) * dC/dZ(3,c))
         */

        for(int i = 1; i <= numNeuronsLayer[1]; i++){
            double curChange = weights[getWeightIndex(place, 3, i)] * getWeightedSumNudgeL3(i);

            totalNudges += curChange;
        }

        return totalNudges;
    }


    /**
     * Returns the desired nudge on a given weighted sum in L2
     * @param place the place of the weighted sum in layer 2
     * @return the desired nudge
     */
    private double getWeightedSumNudgeL2(int place) {
        // add 1 because we are using a zero array in the loop we passed this from
        int neuronIndex = getNeuronIndex(2, place);

        // weighted sum of the neuron we are observing
        double wSum = neuronWeightedSums[neuronIndex];

        // value already calculated
        double preValue = weightedSumNudges[neuronIndex];

        if(preValue != DEFAULT_NUDGE){
            // the only way we are NOT at the default value is we already calculated the necessary nudge.
            // just return that
            return preValue;
        }

        /*
         * Nudging a weighed sum on Layer 2 cannot directly affect the cost function
         *
         * Nudging a weighted sum can directly affect the activation of a layer 2 neuron.
         *
         * The ratio in the change between the change in a last layer activation and a last
         * layer weighted sum is equal to the derivative of the RELU function at the
         * value of the current weighed sum. Therefore, we will multiply the desired
         * activation nudge by said derivative
         */

        weightedSumNudges[neuronIndex] = NeuralMath.reluDeriv(wSum) * getActivationNudgeL2(place);
        return weightedSumNudges[neuronIndex];
    }

    /**
     * Calculates how much we should nudge each bias in layer 3 according to the current dataset
     * @param biasNudges: the array that stores the desired nudges
     */
    private void calculateBiasNudgesL2(double[] biasNudges) {
        for(int i = 1; i <= numNeuronsLayer[1]; i++){
            int index = getBiasIndex(2, i);

            biasNudges[index] = getBiasNudgeL2(i);
        }
    }

    /**
     * returns the desired nudge of a bias in layer 2
     * @param place the place of the neuron in layer 2
     * @return the desired nudge of a bias in layer 2
     */
    private double getBiasNudgeL2(int place) {
        /*
        It's just the nudge of the weighted sum of the neuron this bias is attached to
         */

        return getWeightedSumNudgeL2(place);
    }

    /**
     * Calculates how much we should nudge each weight pointing to  layer 2 according to the current data set
     * @param weightNudge: the array that stores the desired nudges
     */
    private void calculateWeightNudgesL2(double[] weightNudge) {
        // one weight connects every weight in Layer 3 to every neuron in layer 4
        int index;

        for(int startPlace = 1; startPlace <= numNeuronsLayer[0]; startPlace++){ // for every neuron in layer 1
            for(int endPlace = 1; endPlace <= numNeuronsLayer[1]; endPlace++){ // for every neuron in layer 2
                // add 1, because the for loop starts at 0
                index = getWeightIndex(startPlace,2,endPlace);
                weightNudge[index] = getWeightNudgeL2(startPlace,endPlace);
            }
        }
    }

    /**
     * Gets the nudge of a specific weight that points to a neuron in layer 3
     * @param startPlace the place of the starting neuron in layer 2
     * @param endPlace the place of the starting neuron in layer 3
     * @return the desired nudge of the given weight
     */
    private double getWeightNudgeL2(int startPlace, int endPlace) {
        /*
         * Nudging a weight pointing to layer 4 cannot directly affect the cost function.
         *
         * Nudging a weight pointing to layer 4 can affect the weighted sum of its
         * destination neuron.
         *
         * Due to the chain rule, dC/dW(a,2,b) = dz(2,b)/dW(a,2,b) * dC/dZ(2,b)
         *
         * The effect of dW(a,2,b) on dZ(2,b) is equal to the product of dW(b,3,c) and A(1,a).
         * Therefore,the effect of changing dW(a,2,b) is exactly proportional to A(1,a).
         *
         * Therefore, dz(2,b)/dW(a,2,b) = A(2,b)
         */
        double n1Activation = getActivation(1, startPlace);
        double n2weightedSumNudge = getWeightedSumNudgeL2(endPlace);
        return n1Activation * n2weightedSumNudge;
    }

    /**
     * Runs the tests on both the training and testing data
     * @param test the Test implementation that tells us to check if an example is correct
     */
    public void runTests(Test test){
        // the array to store the activations of the final layer
        double[] finalLayerActivation = new double[numNeuronsLayer[LAST_LAYER]];

        // the number of correct examples
        int correct = 0;

        System.out.println("Running Tests for training data");

        for(int i = 0; i < numTrainingExamples; i++){
            runExample(trainingDataInput[i], trainingDataOutput[i]);
            setActivationArray(finalLayerActivation);
            System.out.println(i);
            if(test.runTest(currentDesiredOutput, finalLayerActivation)){
                // correct
                correct++;
            }else{
                // incorrect. Print out the incorrect value
                System.out.println(NeuralMath.printArray(finalLayerActivation) + " is incorrect");
                System.out.println("Expected " + NeuralMath.printArray(currentDesiredOutput));
            }
        }

        System.out.println("Got " + correct + " correct out of " + numTrainingExamples + ", or " + 100.0 * correct/numTrainingExamples + "% of the training data");

        correct = 0;

        System.out.println("Running Tests for testing data");

        for(int i = 0; i < numTestingExamples; i++){
            runExample(testingDataInput[i], testingDataOutput[i]);
            setActivationArray(finalLayerActivation);
            if(test.runTest(currentDesiredOutput, finalLayerActivation)){
                // correct
                correct++;
            }else{
                // incorrect. Print out the incorrect value
                System.out.println(NeuralMath.printArray(finalLayerActivation) + " is incorrect");
                System.out.println("Expected " + NeuralMath.printArray(currentDesiredOutput));
            }
        }

        System.out.println("Got " + correct + " correct out of " + numTestingExamples + ", or " + 100.0 * correct/numTestingExamples + "% of the testing data");
    }


    /**
     * puts the activation of the final layer into an array
     * @param arr the array the activation will be put into
     */
    private void setActivationArray(double @NotNull [] arr)throws IllegalArgumentException{
        int len = arr.length;

        if(len != numNeuronsLayer[LAST_LAYER]){
            throw new IllegalArgumentException("The provided array has " + len +  " spots, while it needs " + numNeuronsLayer[LAST_LAYER]);
        }

        for(int i = 0; i < len; i++){
            arr[i] = getActivation(NUM_LAYERS, i + 1);
        }
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

    public void printAllActivation(){
        for(int i = 1; i <= NUM_LAYERS; i++){
            System.out.print("Layer " + i + " activations: ");
            printActivation(i);
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
        System.out.println();
    }

    /**
     * A function meant for debugging
     */
    public void printAllValues(){
        // change this to choose which data we run
        runExample(trainingDataInput[15],trainingDataOutput[15]);
        resetNudgeArrays();

        double[] weightTest = new double[numWeights];
        double[] biasTest = new double[numBiases];

        calculateWeightNudgesL4(weightTest);
        calculateWeightNudgesL3(weightTest);
        calculateWeightNudgesL2(weightTest);

        calculateBiasNudgesL4(biasTest);
        calculateBiasNudgesL3(biasTest);
        calculateBiasNudgesL2(biasTest);


        //printing weighted sums
        for(int layer = 1; layer <= NUM_LAYERS; layer++){
            for(int place = 1; place <= numNeuronsLayer[layer - 1]; place++){
                System.out.println("The weighted sum of neuron " + place + " on layer " + layer + " is: " + neuronWeightedSums[getNeuronIndex(layer, place)]);
            }
        }

        System.out.println();

        //printing activations:
        printAllActivation();

        double[] testArr = new double[numNeuronsLayer[LAST_LAYER]];
        setActivationArray( testArr);

        System.out.println(NeuralMath.printArray(testArr));

        //printing desired:
        System.out.println("Desired:");
        for(int i = 0; i <  numNeuronsLayer[LAST_LAYER]; i++){
            System.out.print(currentDesiredOutput[i] + " ");
        }

        System.out.println();
        System.out.println();

        for(int layer = 2; layer <= NUM_LAYERS; layer++){
            for(int start = 1; start <= numNeuronsLayer[layer - 2]; start++){
                for(int end = 1; end <= numNeuronsLayer[layer - 1]; end++){
                    System.out.println("The weight that points from neuron " + start + " in layer " + (layer - 1) + " to neuron " + end + " in " + (layer) + ": " + weights[getWeightIndex(start, layer, end)]);
                }
            }
        }

        System.out.println();
        System.out.println("ActivationL4 Nudges");
        for(int i = 1; i <= 3; i++){
            System.out.println(getActivationNudgeL4(i));
        }

        System.out.println();
        System.out.println("WSL4 Nudges");

        //weighted sum L4 nudges
        for(int i = 1; i <=3; i++){
            System.out.println(getWeightedSumNudgeL4(i));
        }

        System.out.println();
        System.out.println("BiasL4 Nudges");

        //weighted sum L4 nudges
        for(int i = 1; i <=3; i++){
            int index = getBiasIndex(4, i);
            System.out.println(biasTest[index]);
        }


        System.out.println();
        System.out.println("L4 weight nudges");
        //weight L4 nudges
        for(int i = 1; i <= 3; i++){
            for(int j = 1; j <= 3; j++){
                int index = getWeightIndex(i,4,j);
                System.out.print(weightTest[index] + " ");
            }
            System.out.println();
        }

        System.out.println();
        System.out.println("ActivationL3 Nudges");
        for(int i = 1; i <=3; i++){
            System.out.println(getActivationNudgeL3(i));
        }

        System.out.println();
        System.out.println("WSL3 Nudges");

        for(int i = 1; i <=3; i++){
            System.out.println(getWeightedSumNudgeL3(i));
        }


        System.out.println();
        System.out.println("BiasL3 Nudges");

        for(int i = 1; i <=3; i++){
            int index = getBiasIndex(3, i);
            System.out.println(biasTest[index]);
        }


        System.out.println();
        System.out.println("L3 weight nudges");
        for(int i = 1; i <= 3; i++){
            for(int j = 1; j <= 3; j++){
                int index = getWeightIndex(i,3,j);
                System.out.print(weightTest[index] + " ");
            }
            System.out.println();
        }


        System.out.println();
        System.out.println("ActivationL2 Nudges");
        for(int i = 1; i <=3; i++){
            System.out.println(getActivationNudgeL2(i));
        }

        System.out.println();
        System.out.println("WSL2 Nudges");

        for(int i = 1; i <=3; i++){
            System.out.println(getWeightedSumNudgeL2(i));
        }

        System.out.println();
        System.out.println("BiasL2 Nudges");

        for(int i = 1; i <=3; i++){
            int index = getBiasIndex(2, i);
            System.out.println(biasTest[index]);
        }

        System.out.println();
        System.out.println("L2 weight nudges");
        for(int i = 1; i <= 3; i++){
            for(int j = 1; j <= 3; j++){
                int index = getWeightIndex(i,2,j);
                System.out.print(weightTest[index] + " ");
            }
            System.out.println();
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
            // we are not on last layer
            activation = NeuralMath.leakyRELU(weightedSum);
        }else{
            // we are on last layer
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