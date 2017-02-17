package mosig.layers;

import mosig.common.ActivationLayer;

public final class ReluActivationLayer extends ActivationLayer {

    public ReluActivationLayer(int n) {
        super(n);
    }

    @Override
    public double activationFunction(double x) {
        return Math.max(0, x);
    }

    @Override
    public double activationDerivative(double x) {
        return x < 0 ? 0 : 1;
    }
}
