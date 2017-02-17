package test;

import mosig.common.NetworkBuffer;
import mosig.common.Util;
import mosig.layers.FullyConnectedLayer;
import mosig.layers.SigmoidActivationLayer;

public class DebugCounting {

    public static final class NeuralNet {
        private NetworkBuffer buffer0 = new NetworkBuffer(3);

        private FullyConnectedLayer layer1 = new FullyConnectedLayer(3, 4);
        private NetworkBuffer buffer1 = new NetworkBuffer(4);
        private SigmoidActivationLayer layer2 = new SigmoidActivationLayer(4);
        private NetworkBuffer buffer2 = new NetworkBuffer(4);

        private FullyConnectedLayer layer3 = new FullyConnectedLayer(4, 3);
        private NetworkBuffer buffer3 = new NetworkBuffer(3);
        private SigmoidActivationLayer layer4 = new SigmoidActivationLayer(3);
        private NetworkBuffer buffer4 = new NetworkBuffer(3);

        public NeuralNet() {
            layer1.sharedInput = buffer0;
            layer1.sharedOutput = buffer1;
            layer2.sharedInput = buffer1;
            layer2.sharedOutput = buffer2;
            layer3.sharedInput = buffer2;
            layer3.sharedOutput = buffer3;
            layer4.sharedInput = buffer3;
            layer4.sharedOutput = buffer4;
            Util.initWeights(layer1);
            Util.initWeights(layer2);
            Util.initWeights(layer3);
            Util.initWeights(layer4);
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
            layer2.forward();
            layer3.forward();
            layer4.forward();
        }

        private double backward(double[] output) {
            double lastError = buffer4.initGradientL2(output);
            layer4.backward();
            layer3.backward();
            layer2.backward();
            layer1.backward();
            return lastError;
        }

        private void adjust(double learningRate) {
            layer1.adjustWeights(learningRate);
            layer2.adjustWeights(learningRate);
            layer3.adjustWeights(learningRate);
            layer4.adjustWeights(learningRate);
        }
    }

    public static void main(String[] args) {
        NeuralNet nn = new NeuralNet();

        double[][] inputs = new double[][] {
                {0, 0, 0},
                {0, 0, 1},
                {0, 1, 0},
                {0, 1, 1},
                {1, 0, 0},
                {1, 0, 1},
                {1, 1, 0},
                {1, 1, 1},
        };

        double[][] outputs = new double[][] {
                {0, 0, 1},
                {0, 1, 0},
                {0, 1, 1},
                {1, 0, 0},
                {1, 0, 1},
                {1, 1, 0},
                {1, 1, 1},
                {0, 0, 0},
        };

        for (int i = 0; i < 10000000; i++) {
            double lastError = 0;
            for (int j = 0; j < 1; j++) {
                lastError = nn.train(inputs[i % 8], outputs[i % 8], 0.1);
            }

            if (i % 1000001 == 0) {
                i = 0;
                System.out.println(lastError);
            }
        }
    }
}
