package test;

import mosig.common.NetworkBuffer;
import mosig.common.Util;
import mosig.layers.FullyConnectedLayer;
import mosig.layers.SigmoidActivationLayer;

public class DebugXor {

    public static final class NeuralNet {
        // input 2
        NetworkBuffer buffer0 = new NetworkBuffer(2);

        // hidden 2
        FullyConnectedLayer layer1 = new FullyConnectedLayer(2, 2);
        NetworkBuffer buffer1 = new NetworkBuffer(2);
        SigmoidActivationLayer layer2 = new SigmoidActivationLayer(2);
        NetworkBuffer buffer2 = new NetworkBuffer(2);

        // hidden 2
        FullyConnectedLayer layer3 = new FullyConnectedLayer(2, 2);
        NetworkBuffer buffer3 = new NetworkBuffer(2);
        SigmoidActivationLayer layer4 = new SigmoidActivationLayer(2);
        NetworkBuffer buffer4 = new NetworkBuffer(2);

        // sharedOutput 1
        FullyConnectedLayer layer5 = new FullyConnectedLayer(2, 1);
        NetworkBuffer buffer5 = new NetworkBuffer(1);
        SigmoidActivationLayer layer6 = new SigmoidActivationLayer(1);
        NetworkBuffer buffer6 = new NetworkBuffer(1);


        public NeuralNet() {
            layer1.sharedInput = buffer0;
            layer1.sharedOutput = buffer1;
            layer2.sharedInput = buffer1;
            layer2.sharedOutput = buffer2;
            layer3.sharedInput = buffer2;
            layer3.sharedOutput = buffer3;
            layer4.sharedInput = buffer3;
            layer4.sharedOutput = buffer4;
            layer5.sharedInput = buffer4;
            layer5.sharedOutput = buffer5;
            layer6.sharedInput = buffer5;
            layer6.sharedOutput = buffer6;
            Util.initWeights(layer1);
            Util.initWeights(layer2);
            Util.initWeights(layer3);
            Util.initWeights(layer4);
            Util.initWeights(layer5);
            Util.initWeights(layer6);
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
            layer5.forward();
            layer6.forward();
        }

        private double backward(double[] output) {
            double lastError = buffer6.initGradientL2(output);
            layer6.backward();
            layer5.backward();
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
            layer5.adjustWeights(learningRate);
            layer6.adjustWeights(learningRate);
        }
    }

    public static void main(String[] args) {
        NeuralNet nn = new NeuralNet();

        double[][] inputs = {
                {0, 0},
                {0, 1},
                {1, 0},
                {1, 1}
        };

        double[][] outputs = {
                {0},
                {1},
                {1},
                {0},
        };

        for (int i = 0; i < 100000000; i++) {
            double lastError = nn.train(inputs[i % 4], outputs[i % 4], 0.01);

            if (i % 100001 == 0) {
                System.out.println(lastError);
            }
        }

    }

}
