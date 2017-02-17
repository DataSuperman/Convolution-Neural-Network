package mosig.layers;

import mosig.common.ActivationLayer;
import mosig.common.Util;

public final class LeakyReluActivationLayer extends ActivationLayer {

    public LeakyReluActivationLayer(int n) {
        super(n);
    }

    @Override
    public double activationFunction(double x) {
        return Math.max(0.01 * x, x);
    }

    @Override
    public double activationDerivative(double x) {
        return x < 0 ? 0.01 : 1;
    }

    @Override
    public void forward() {
        double[] inputValues = sharedInput.values;
        double[] outputValues = sharedOutput.values;

        for (int i = 0; i < layerSize; i++) {
            double x = inputValues[i];
            outputValues[i] = Math.max(0.01 * x, x);
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
            inputGradient[i] = (inputValues[i] < 0 ? 0.01 : 1) * outputGradient[i];
        }
    }

}
