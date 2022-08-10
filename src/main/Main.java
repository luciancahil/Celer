package main;
import main.CelerNetwork.NeuralMath.Test;
import main.CelerNetwork.NeuralNetwork;
import main.CelerNetwork.NeuralMath.NeuralMath;


import java.io.*;
import java.util.Random;


public class Main {
    public static void main(String[] args) throws IOException {

        int inputSize = 784;
        int outputSize = 10;
        int numSamples = 60000;

       // int inputSize = 3;
      //  int outputSize = 3;
      //  int numSamples = 1000;
        //Random generator = new Random(5361884675712300576L);

        double[][] inputArray = new double[numSamples][inputSize];
        double[][]  outputArray = new double[numSamples][outputSize];


        setData("data/mnist_digits.csv", inputArray, outputArray);

        /*
        for(int i = 0; i < numSamples; i++){
            for(int j = 0; j < inputSize; j++){
                inputArray[i][j] = generator.nextDouble();
                outputArray[i][j] = generator.nextDouble();
            }
        }*/

        NeuralNetwork network = new NeuralNetwork(inputSize, outputSize,5361884675712300576L);
        TestImplementation test = new TestImplementation();


        network.setData(inputArray,outputArray);
        //network.printAllValues();
        network.train();
        network.runTests(test);
        network.printNetwork();
    }

    /**
     *
     * @param filePath - The path to the data file we use as data
     * @param inputArray - The array we place all input values into
     * @param outputArray - The array we place all output values into
     * @throws IOException: When the filepath does not lead to a valid path
     */
    private static void setData(String filePath, double[][] inputArray, double[][] outputArray) throws IOException {
        // file reader that will read the data file
        FileReader data = new FileReader(filePath);

        // BufferedReader that will read the data file
        BufferedReader bufferedRead = new BufferedReader(data);

        // string that contains a single line of data
        String nextLine = bufferedRead.readLine();

        // index of the next comma in nextLine
        int commaIndex;

        // index of the previous comma in the next line
        int oldCommaIndex;

        // the number between commas we will parse
        String numberString;

        // the actual integer we parsed out of the data
        int dataPoint;

        // run through each line in the file
        for(int i = 0; i < 60000; i++){

            // start looking for commas at the beginning of the file
            oldCommaIndex = -1;

            /**
             * Run through each number in the line.
             * Each number is separated by a comma.
             * The first number represents what number was hand drawn.
             * The next 784 numbers represent the brightness of a given pixel
             * in a 28 x 28 display. 255 is fully activated, 0 is not activated
             */
            for(int j = 0; j < 784; j++){

                // locate the next comma
                commaIndex = nextLine.indexOf(",", oldCommaIndex + 1);

                // get a string that is purely a number, without any commas
                if(commaIndex != -1) {
                    // there is at least one more comma in the file
                    numberString = nextLine.substring(oldCommaIndex + 1, commaIndex);
                }else{
                    // there are no more commas in the file. We have reached the final number
                    numberString = nextLine.substring(oldCommaIndex + 1);
                }

                // get an actual int data type from the number string
                dataPoint = Integer.parseInt(numberString);

                if(j == 0){
                    // this represents the label
                    outputArray[i][dataPoint] = 1;
                }else{
                    // this number represents an activation
                    // we divide by 256 in order to get a number between 0 and 10,
                    // to make calibration of the neural network easier
                    inputArray[i][j] = dataPoint/25.6;
                }
                oldCommaIndex = commaIndex;
            }
            nextLine = bufferedRead.readLine();
        }


    }

    /**
     *
     * @param image
     */
    private static void showImage(double[] image){
        for(int i = 0; i < 28; i++){
            for(int j = 0; j < 28; j++){
                double intensity = image[28 * i + j];
                if(intensity ==0){
                    System.out.print("  ");
                }else if(intensity < 180/255.0){
                    System.out.print(".");
                }else{
                    System.out.print(" @");
                }
            }
            System.out.println();
        }
    }
}


class TestImplementation implements Test{
    @Override
    public boolean runTest(double[] expected, double[] actual) {
        return runSelectionTest(expected, actual);
    }

    /**
     * A test to run if only one neuron is "active"
     * @param expected the array that contains the expected actiavtion of the final layer
     * @param actual the actual final layer of a neurla netwrok
     * @return true if the most active neuron is as expected, false otherwise
     */
    private boolean runSelectionTest(double[] expected, double[] actual) throws IllegalArgumentException{
        // the index of the neuron that is "active".
        int activatedIndex = -1;
        int len = expected.length;
        // the largest neuron in the actual array
        int mostActiveIndex = 0;

        for(int i = 0; i < len; i++){

            if(expected[i] == 1){
                // we have found the correct index of the neuron to turn on
                activatedIndex = i;
                break;
            }
        }

        if(activatedIndex == -1){
            // no active neuron was found

            throw new IllegalArgumentException("There was no avtive neuron.");
        }

        // find the most active index in the actual array
        for(int i = 1; i < len; i++){
            if(actual[i] > actual[mostActiveIndex]){
                // we have found a more active neuron
                mostActiveIndex = i;
            }
        }

        // is the most activated index the one we wanted?
        return (mostActiveIndex == activatedIndex);
    }
    
    /**
     * A test to run if only many neurons are active, and we need to check the correct ones are
     * @param expected the array that contains the expected actiavtion of the final layer
     * @param actual the actual final layer of a neurla netwrok
     * @return true if the correct neurons are on.
     */
    private boolean runTrueFalseTest(double[] expected, double[] actual) throws IllegalArgumentException{
        // the index of the neuron that is "active".
        int len = expected.length;


        for(int i = 0; i < len; i++){
            if(expected[i] == 1 && actual[i] <= 0.5){
                // the neuron is supposed to be on, but isn't
                return false;
            } else if (expected[i] == 0 && actual[i] > 0.5) {
                // the neuron is supposed to be off, but is on.
                return false;
            }
        }

        // if we get here, we haven't found a mistake. Thus, our neurons are Correct
        return true;
    }
}
