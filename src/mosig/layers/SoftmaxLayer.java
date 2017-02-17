package mosig.layers;

import mosig.common.Layer;
import mosig.common.NetworkBuffer;
import mosig.common.Util;

import java.io.PrintStream;
import java.util.Scanner;

public class SoftmaxLayer extends Layer {
    public NetworkBuffer sharedInput;
    public NetworkBuffer sharedOutput;

    public void forward() {
        double[] input = sharedInput.values;
        double[] output = sharedOutput.values;
        assert input.length == output.length;
        int n = input.length;

        double max = input[0];
        for (int i = 1; i < n; i++) {
            if (max < input[i]) {
                max = input[i];
            }
        }

        double sum = 0;
        for (int i = 0; i < n; i++) {
            double e = Math.exp(input[i] - max);
            sum += e;
            output[i] = e;
        }

        double invSum = 1 / sum;
        for (int i = 0; i < n; i++) {
            output[i] *= invSum;
        }
    }

    public double backward(int y) {
        double[] outputValues = sharedOutput.values;
        double[] inputGradients = sharedInput.gradients;
        assert inputGradients.length == outputValues.length;

        int n = inputGradients.length;
        Util.setToZero(inputGradients);

        for (int i = 0; i < n; i++) {
            double indicator = (i == y ? 1 : 0);
            inputGradients[i] = -(indicator - outputValues[i]);
        }

        return -Math.log(outputValues[y]);
    }

    public void adjustWeights(double learningRate) {
    }

    public void readWeightsFrom(Scanner input) {
    }

    public void writeWeightsTo(PrintStream output) {
    }

}
