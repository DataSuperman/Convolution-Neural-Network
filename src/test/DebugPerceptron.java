package test;

import mosig.common.NetworkBuffer;
import mosig.common.Util;
import mosig.layers.FullyConnectedLayer;

public class DebugPerceptron {

    public static final class NeuralNet {
        final NetworkBuffer buffer0 = new NetworkBuffer(2);
        final FullyConnectedLayer layer1 = new FullyConnectedLayer(2, 1);
        final NetworkBuffer buffer1 = new NetworkBuffer(1);

        NeuralNet() {
            layer1.sharedInput = buffer0;
            layer1.sharedOutput = buffer1;
            Util.initWeights(layer1);
        }

        public double train(double[] input, double[] output, double learningRate) {
            forward(input);
            double lastError = backward(output);
            adjust(learningRate);
            return lastError;
        }

        private void forward(double[] input) {
            Util.copy(input, buffer0.values);
            layer1.forward();
        }

        private double backward(double[] output) {
            double lastError = buffer1.initGradientL2(output);
            layer1.backward();
            return lastError;
        }

        private void adjust(double learningRate) {
            layer1.adjustWeights(learningRate);
        }
    }

    public static void main(String[] args) {
        double input1, input2, expect1;
        double[] inputBuffer = new double[2];
        double[] expectedOutput = new double[1];

        NeuralNet nn = new NeuralNet();

        for (int i = 0; i < 100000000; i++) {
            input1 = Math.random() * 100;
            input2 = Math.random() * 100;
            expect1 = 0.5f * input1 + 0.5f  * input2; // the formula the network has to guess

            inputBuffer[0] = input1;
            inputBuffer[1] = input2;
            expectedOutput[0] = expect1;

            double lastError = nn.train(inputBuffer, expectedOutput, 0.0001);

            if (i % 100001 == 0) {
                System.out.println(lastError);
            }
        }

    }
}
