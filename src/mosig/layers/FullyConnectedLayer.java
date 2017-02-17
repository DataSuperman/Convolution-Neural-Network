package mosig.layers;

import mosig.common.Layer;
import mosig.common.NetworkBuffer;
import mosig.common.Util;

import java.io.PrintStream;
import java.util.Scanner;

public final class FullyConnectedLayer extends Layer {
    public final int inputSize;
    public final int outputSize;

    public final double[][] weights;
    public final double[] bias;

    public final double[][] weightGradients;
    public final double[] biasGradients;

    public NetworkBuffer sharedInput;
    public NetworkBuffer sharedOutput;

    private final double[] inputBuffer;
    private final double[] outputBuffer;
    private final double[] inputGradient;
    private final double[] outputGradient;

    public FullyConnectedLayer(int inputSize, int outputSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;

        this.weights = new double[outputSize][inputSize];
        this.bias = new double[outputSize];

        this.weightGradients = new double[outputSize][inputSize];
        this.biasGradients = new double[outputSize];

        inputBuffer = new double[inputSize];
        outputBuffer = new double[outputSize];
        inputGradient = new double[inputSize];
        outputGradient = new double[outputSize];
    }

    @Override
    public void readWeightsFrom(Scanner input) {
        for (int i = 0; i < outputSize; i++) {
            bias[i] = input.nextDouble();
            double[] w = weights[i];
            for (int j = 0; j < inputSize; j++) {
                w[j] = input.nextDouble();
            }
        }
    }

    @Override
    public void writeWeightsTo(PrintStream output) {
        for (int i = 0; i < outputSize; i++) {
            output.print(bias[i]);
            output.print(' ');
            double[] w = weights[i];
            for (int j = 0; j < inputSize; j++) {
                output.print(w[j]);
                output.print(' ');
            }
            output.print('\n');
        }
    }

    public void forward() {
        assert sharedInput.size == inputSize;
        assert sharedOutput.size == outputSize;

        Util.copy(sharedInput.values, inputBuffer);
        {
            for (int i = 0; i < outputSize; i++) {
                double total = bias[i];
                double[] w = weights[i];
                for (int j = 0; j < inputSize; j++) {
                    total += w[j] * inputBuffer[j];
                }
                outputBuffer[i] = total;
            }
        }
        Util.copy(outputBuffer, sharedOutput.values);
    }

    public void backward() {
        assert sharedInput.size == inputSize;
        assert sharedOutput.size == outputSize;

        Util.copy(sharedOutput.gradients, outputGradient);
        {
            Util.setToZero(inputGradient);
            for (int i = 0; i < outputSize; i++) {
                double dE = outputGradient[i];
                double[] w = weights[i];
                double[] dw = weightGradients[i];
                for (int j = 0; j < inputSize; j++) {
                    inputGradient[j] += dE * w[j];
                    dw[j] += inputBuffer[j] * dE;
                }
                biasGradients[i] += dE;
            }
        }
        Util.copy(inputGradient, sharedInput.gradients);
    }

    public void adjustWeights(double learningRate) {
        assert sharedInput.size == inputSize;
        assert sharedOutput.size == outputSize;

        for (int i = 0; i < outputSize; i++) {
            double[] w = weights[i];
            double[] dw = weightGradients[i];

            for (int j = 0; j < inputSize; j++) {
                w[j] -= learningRate * dw[j];
            }
            bias[i] -= learningRate * biasGradients[i];
        }

        Util.setToZero(weightGradients);
        Util.setToZero(biasGradients);
    }

}
