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
            return input * LEAKY_COEFFICIENT;
        }
    }

    /**
     * Returns the derivative of the RELU function based on a given input
     * @param input the input to the RELU function
     * @return derivative of the RELU function
     */
    public static double reluDeriv(double input){
        if(input >= 0){
            return 1;
        }else{
            return LEAKY_COEFFICIENT;
        }
    }


    /**
     * Returns the sigmoid of a given input. The sigmoid function s(x) is equal to 1 / (1 + e^-x).
     * It will return a number between 0 and 1 for all real numbers, and the output wll be
     * closer to 0 for smaller inputs, and closer to 1 for larger inputs.
     * @param input the number we are calculating the sigmoid of
     * @return The sigmoid function of input (always a number between 0 and 1)
     */
    public static double sigmoid(double input){
        return 1.0 / (1 + Math.pow(Math.E, input * - 1));
    }


    public static double sigmoidDeriv(double input){
        double exp = Math.exp(input);

        if(Double.isInfinite(exp)){
            return 0;
        }

        return exp / Math.pow(1 + exp, 2);
    }


    /**
     * Updates the avgs array by adding newVal to it.
     *
     * @param avgs the array containg a running average
     * @param newVals the values we would like to include in the avgs
     * @param count the number of values the average has, including the new values
     */
    public static void updateRollingAvgs(double[] avgs, double[] newVals, int count){
        if(avgs.length != newVals.length){
            throw new IllegalArgumentException("Avgs and NewVals must be same-sized arrays");
        }

        for(int i = 0; i < avgs.length; i++){
            avgs[i] = updateRollingAvg(avgs[i], newVals[i], count);
        }
    }

    /**
     * Returns the new rolling average by adding newVal to the average
     * @param avg the rolling average without newVal
     * @param newVal the value we wish to add
     * @param count the number of values the rolling average contains, including newVal
     * @return
     */
    public static double updateRollingAvg(double avg, double newVal, int count) {
        if(count == 1 && avg != 0){
            throw new IllegalArgumentException("There are no previous elements, so the average should be 0");
        }

        /*
         * Proof this works:
         *
         * The sum of all values without newVal is equal to avg * (count - 1) = avg * count - avg
         *
         * The sum including newVal is avg * count - avg + newVal
         *
         * The new average is (avg * count - avg + newVal) / count
         *
         * Dividing through ends up with avg - avg / count + newVal / count
         */
        return avg - avg/count + newVal/count;
    }

    /**
     * Prints a given array to the command line
     * @param arr the array to print
     */
    public static String printArray(double[] arr){
        int len = arr.length;
        StringBuilder result = new StringBuilder();

        if (len == 0) {
            return "{}";
        }

        result.append("{" + arr[0]);

        for(int i = 1; i < len; i++){
            result.append(", " + arr[i]);
        }

        result.append("}");

        return result.toString();
    }
}
