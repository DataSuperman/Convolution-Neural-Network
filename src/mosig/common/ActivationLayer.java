package mosig.common;

import java.io.PrintStream;
import java.util.Scanner;

public abstract class ActivationLayer extends Layer {
    public final int layerSize;
    public NetworkBuffer sharedInput;
    public NetworkBuffer sharedOutput;

    public ActivationLayer(int n) {
        layerSize = n;
    }

    public abstract double activationFunction(double x);

    public abstract double activationDerivative(double x);

    @Override
    public void forward() {
        double[] inputValues = sharedInput.values;
        double[] outputValues = sharedOutput.values;

        for (int i = 0; i < layerSize; i++) {
            outputValues[i] = activationFunction(inputValues[i]);
        }
    }

    @Override
    public void backward() {
        assert sharedInput.size == layerSize;
        assert sharedOutput.size == layerSize;

        double[] inputValues = sharedInput.values;
        double[] inputGradient = sharedInput.gradients;
        double[] outputGradient = sharedOutput.gradients;

        Util.setToZero(inputGradient);
        for (int i = 0; i < layerSize; i++) {
            inputGradient[i] = activationDerivative(inputValues[i]) * outputGradient[i];
        }
    }

    @Override
    public final void adjustWeights(double learningRate) {
    }

    @Override
    public final void readWeightsFrom(Scanner input) {
    }

    @Override
    public final void writeWeightsTo(PrintStream output) {
    }
}
