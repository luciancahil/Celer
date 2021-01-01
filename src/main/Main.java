package main;
import main.CelerNetwork.NeuralNetwork;


import java.io.*;
//TODO add documentation to setData


public class Main {
    public static void main(String[] args) throws IOException {
        double[][] inputArray = new double[60000][784];
        double[][]  outputArray = new double[60000][10];

        setData("data/mnist_digits.csv", inputArray, outputArray);

        for(int i = 0; i < 256; i++){
            System.out.println(inputArray[12][i] + " vs. " + inputArray[12][i]*256);
        }
        showImage(inputArray[1]);
    }


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
        for(int j = 0; j < 60000; j++){

            // start looking for commas at the begining of the file
            oldCommaIndex = -1;

            /**
             * Run through each number in the line.
             * Each number is seperated by a comma.
             * The first number represents what number was hand drawn.
             * The next 784 numbers represent the brightness of a given pixel
             * in a 28 x 28 display. 255 is fully activated, 0 is not activated
             */
            for(int i = 0; i < 784; i++){

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

                if(i == 0){
                    // this represents the label
                    outputArray[j][dataPoint] = 1;
                }else{
                    inputArray[j][i] = dataPoint/256.0;
                }
                oldCommaIndex = commaIndex;
            }
        }


    }

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
