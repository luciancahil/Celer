package main;
import main.CelerNetwork.NeuralNetwork;


import java.io.*;

public class Main {
    public static void main(String[] args) throws IOException {
        double[][] inputArray = new double[60000][784];
        double[][]  outputArray = new double[60000][10];

        //setData("data/mnist_digits.csv", inputArray, outputArray);

        NeuralNetwork network = new NeuralNetwork(10, 40);
    }

    /**
     *
     * @param fileName - The path to the data file we use as data
     * @param inputArray - The array we place all input values into
     * @param outputArray - The array we place all output values into
     * @throws IOException: When the filepath does not lead to a valid path
     */
    private static void setData(String fileName, double[][] inputArray, double[][] outputArray) throws IOException {
        // file reader that will read the data file
        FileReader data = new FileReader(fileName);

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
                    // we divide by 256 in order to get a number between 0 and 1,
                    // to make calibration of the neural network easier
                    inputArray[i][j] = dataPoint/256.0;
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
