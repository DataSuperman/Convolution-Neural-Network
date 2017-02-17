package mosig.layers;

import mosig.common.ActivationLayer;
import mosig.common.Util;

public final class SigmoidActivationLayer extends ActivationLayer {

    public SigmoidActivationLayer(int n) {
        super(n);
    }

    public static double f(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    public static double df(double x) {
        double f_x = f(x);
        return f_x * (1 - f_x);
    }

    public double activationFunction(double x) {
        return f(x);
    }

    public double activationDerivative(double x) {
        return df(x);
    }

    @Override
    public void backward() {
        assert sharedInput.size == layerSize;
        assert sharedOutput.size == layerSize;

        double[] inputGradient = sharedInput.gradients;
        double[] outputGradient = sharedOutput.gradients;
        double[] outputValues = sharedOutput.values;

        Util.setToZero(inputGradient);
        for (int i = 0; i < layerSize; i++) {
            double f_x = outputValues[i];
            double df_x = f_x * (1 - f_x);
            inputGradient[i] = df_x * outputGradient[i];
        }
    }

}
