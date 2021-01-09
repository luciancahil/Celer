package main.CelerNetwork.NeuralMath;

public class NeuralMath {
    /* Coefficient that will be used to calculate the output of the relu function when a number is less than 0 */
    public static final double LEAKY_COEFFICIENT = 0.01;


    /**
     * A leaky Rectified Linear Unit formula. A simple formula that returns the input if said input is
     * greater than or equal to 0, and returns the input times a small number if the input is less than 0
     * @param input: the number we will calculate the relu of
     * @return the input as is if the input is greater than or equal to 0 or the input times LEAKY_COEFFICIENT if the input is negative
     */
    public static double leakyRELU(double input){
        if(input >= 0){
            return input;
        }else{
            return input *= LEAKY_COEFFICIENT;
        }
    }


    /**
     * Returns the sigmoid of a given input. The sigmoid function s(x) is equal to 1 / (1 + e^-x).
     * It will return a number between 0 and 1 for all real numbers, and the output wll be
     * closer to 0 for smaller inputs, and closer to 1 for larger inputs.
     * @param input
     * @return
     */
    public static double sigmoid(double input){
        return 1.0 / (1 + Math.pow(Math.E, input * - 1));
    }

}
