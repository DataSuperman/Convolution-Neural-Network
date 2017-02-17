package mosig.common;

/**
 * A flat buffer used for storing intermediate results in a neural net.
 */
public final class NetworkBuffer {
    public final double[] values;
    public final double[] gradients;
    public final int size;

    public NetworkBuffer(int size) {
        this.size = size;
        this.values = new double[size];
        this.gradients = new double[size];
    }

    public double initGradientL1(double[] expected) {
        assert expected.length == values.length;

        Util.setToZero(gradients);
        double totalError = 0;
        for (int i = 0; i < expected.length; i++) {
            double dE = values[i] - expected[i];
            totalError += Math.abs(dE);
            gradients[i] = (dE > 0) ? 1 : -1;
        }

        return totalError;
    }

    public double initGradientL2(double[] expected) {
        assert expected.length == values.length;

        Util.setToZero(gradients);
        double totalError = 0;
        for (int i = 0, n = expected.length; i < n; i++) {
            double dE = values[i] - expected[i];
            totalError += dE * dE;
            gradients[i] = dE;
        }

        return totalError;
    }
}
