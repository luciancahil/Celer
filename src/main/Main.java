package main;
import main.CelerNetwork.NeuralNetwork;


public class Main {
    public static void main(String[] args) {
        NeuralNetwork network = new NeuralNetwork(256, 10, 8379480594276100096L);
        System.out.println(network.getNumNeurons());
        System.out.println(network.getNumWeights() + network.getNumBiases());
        System.out.println(network.getWeight(14));
        System.out.println(network.getBias(12));
    }
}
