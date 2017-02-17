package test;

import mosig.common.*;
import mosig.layers.RgbConvolutionLayer;

import java.io.IOException;

public class DebugRgbConvolution {

    public static final class NeuralNet {
        NetworkBuffer buffer0 = new NetworkBuffer(32 * 32 * 3);
        RgbConvolutionLayer layer1 = new RgbConvolutionLayer(32, 32, 1, 5, 5, 1);
        NetworkBuffer buffer1 = new NetworkBuffer(32 * 32);

        public NeuralNet() {
            layer1.sharedInput = buffer0;
            layer1.sharedOutput = buffer1;
            Util.initWeights(layer1);
        }

        public int i = 0;

        public double train(RgbImage input, Image expect, double learningRate) {
            forward(input);
            double lastError = backward(expect.data);
            adjust(learningRate);
            return lastError;
        }

        public void forward(RgbImage input) {
            Util.copy(input, buffer0.values);
            layer1.forward();
        }

        public double backward(double[] expect) {
            double lastError = buffer1.initGradientL1(expect);
            layer1.backward();
            return lastError;
        }

        public void adjust(double learningRate) {
            if (i++ >= 5) {
               layer1.adjustWeights(learningRate);
               i = 0;
            }
        }
    }

    public static void main(String[] args) throws IOException {
        NeuralNet nn = new NeuralNet();
        RgbKernel guessMe = new RgbKernel(5, 5);
        double bias = Math.random();

        Util.initUniform(guessMe.r.weights, -1, 1);
        Util.initUniform(guessMe.g.weights, -1, 1);
        Util.initUniform(guessMe.b.weights, -1, 1);

        RgbImage input = new RgbImage(32, 32);
        Image expect = new Image(32, 32);

        for (int i = 0; i < 100000000; i++) {
            //Util.copy(ds.X[i % 5000], input);
            Util.initGaussian(input.r.data, 0, 1);
            Util.initGaussian(input.g.data, 0, 1);
            Util.initGaussian(input.b.data, 0, 1);
            expect.setToConstant(bias);
            guessMe.convolution(input, expect);

            double lastError = nn.train(input, expect, 0.01 / nn.buffer1.size);

            if (i % 1001 == 0) {
                double totalError = 0;
                System.out.println("------------------------------");
                for (int y = 0; y < 5; y++) {
                    for (int x = 0; x < 5; x++) {
                        double e = guessMe.r.getWeight(x, y) - nn.layer1.kernels[0].r.getWeight(x, y);
                        System.out.print(e);
                        System.out.print(x == 4 ? '\n' : "   ");
                        totalError += Math.abs(e);
                    }

                    for (int x = 0; x < 5; x++) {
                        double e = guessMe.g.getWeight(x, y) - nn.layer1.kernels[0].g.getWeight(x, y);
                        System.out.print(e);
                        System.out.print(x == 4 ? '\n' : "   ");
                        totalError += Math.abs(e);
                    }

                    for (int x = 0; x < 5; x++) {
                        double e = guessMe.b.getWeight(x, y) - nn.layer1.kernels[0].b.getWeight(x, y);
                        System.out.print(e);
                        System.out.print(x == 4 ? "\n\n" : "   ");
                        totalError += Math.abs(e);
                    }
                }
                totalError += bias - nn.layer1.biases[0];
                System.out.println(bias - nn.layer1.biases[0]);
                System.out.println("TOTAL ERROR: " + totalError);
                System.out.println("TRAINING ERROR: " + lastError);
                System.out.println("------------------------------");
            }

        }

    }

}
