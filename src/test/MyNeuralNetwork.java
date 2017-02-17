package test;

import mosig.common.Layer;
import mosig.common.NetworkBuffer;
import mosig.common.Util;
import mosig.layers.*;

import java.io.*;
import java.util.Scanner;

public final class MyNeuralNetwork {

    // 32x32xRGB
    public final NetworkBuffer buffer0 = new NetworkBuffer(32 * 32 * 3);

    // 32x32xRGB [*] 5x5xRGB(x16) -> 32x32(x16)
    public final RgbConvolutionLayer layer1 = new RgbConvolutionLayer(32, 32, 1, 5, 5, 16);
    public final NetworkBuffer buffer1 = new NetworkBuffer(32 * 32 * 16);

    // 32x32(x16) -> pool(2,2) -> leaky ReLU -> 16x16(x16)
    public final PoolingLayer layer2 = new PoolingLayer(32, 32, 16, 2, 2);
    public final NetworkBuffer buffer2 = new NetworkBuffer(16 * 16 * 16);
    public final LeakyReluActivationLayer layer3 = new LeakyReluActivationLayer(16 * 16 * 16);
    public final NetworkBuffer buffer3 = new NetworkBuffer(16 * 16 * 16);

    // 16x16(x16) [*] 5x5(x32) -> 16x16(x32)
    public final ConvolutionLayer layer4 = new ConvolutionLayer(16, 16, 16, 5, 5, 32);
    public final NetworkBuffer buffer4 = new NetworkBuffer(16 * 16 * 32);

    // 16x16(x32) -> pool(2,2) -> leaky ReLU -> 8x8(x32)
    public final PoolingLayer layer5 = new PoolingLayer(16, 16, 32, 2, 2);
    public final NetworkBuffer buffer5 = new NetworkBuffer(8 * 8 * 32);
    public final LeakyReluActivationLayer layer6 = new LeakyReluActivationLayer(8 * 8 * 32);
    public final NetworkBuffer buffer6 = new NetworkBuffer(8 * 8 * 32);

    // 8x8(x32) [*] 5x5(x16) -> 8x8(x16)
    public final ConvolutionLayer layer7 = new ConvolutionLayer(8, 8, 32, 5, 5, 16);
    public final NetworkBuffer buffer7 = new NetworkBuffer(8 * 8 * 16);

    // 8x8(x16) -> pool(2,2) -> leaky ReLU -> 4x4(x16)
    public final PoolingLayer layer8 = new PoolingLayer(8, 8, 16, 2, 2);
    public final NetworkBuffer buffer8 = new NetworkBuffer(4 * 4 * 16);
    public final LeakyReluActivationLayer layer9 = new LeakyReluActivationLayer(4 * 4 * 16);
    public final NetworkBuffer buffer9 = new NetworkBuffer(4 * 4 * 16);

    // 4x4x16(=256) -> 10
    public final FullyConnectedLayer layer10 = new FullyConnectedLayer(4 * 4 * 16, 10);
    public final NetworkBuffer buffer10 = new NetworkBuffer(10);

    // sigmoid
    public final SigmoidActivationLayer layer11 = new SigmoidActivationLayer(10);
    public final NetworkBuffer buffer11 = new NetworkBuffer(10);

    public MyNeuralNetwork() {
        layer1.sharedInput = buffer0;
        layer2.sharedInput = buffer1;
        layer3.sharedInput = buffer2;
        layer4.sharedInput = buffer3;
        layer5.sharedInput = buffer4;
        layer6.sharedInput = buffer5;
        layer7.sharedInput = buffer6;
        layer8.sharedInput = buffer7;
        layer9.sharedInput = buffer8;
        layer10.sharedInput = buffer9;
        layer11.sharedInput = buffer10;

        layer1.sharedOutput = buffer1;
        layer2.sharedOutput = buffer2;
        layer3.sharedOutput = buffer3;
        layer4.sharedOutput = buffer4;
        layer5.sharedOutput = buffer5;
        layer6.sharedOutput = buffer6;
        layer7.sharedOutput = buffer7;
        layer8.sharedOutput = buffer8;
        layer9.sharedOutput = buffer9;
        layer10.sharedOutput = buffer10;
        layer11.sharedOutput = buffer11;

        Util.initWeights(layer1);
        Util.initWeights(layer2);
        Util.initWeights(layer3);
        Util.initWeights(layer4);
        Util.initWeights(layer5);
        Util.initWeights(layer6);
        Util.initWeights(layer7);
        Util.initWeights(layer8);
        Util.initWeights(layer9);
        Util.initWeights(layer10);
        Util.initWeights(layer11);

    }

    public void loadFromFolder(String weightsPath) {
        File weightsFolder = new File(weightsPath);
        if (weightsFolder.exists()) {
            loadLayer(weightsFolder, "layer1.txt", layer1);
            loadLayer(weightsFolder, "layer4.txt", layer4);
            loadLayer(weightsFolder, "layer7.txt", layer7);
            loadLayer(weightsFolder, "layer10.txt", layer10);
        }
    }

    private void loadLayer(File weightsFolder, String fileName, Layer layer) {
        File file = new File(weightsFolder, fileName);
        try(Scanner input = new Scanner(new FileInputStream(file))) {
            layer.readWeightsFrom(input);
        } catch (Exception ignored) {
        }
    }

    public void saveToFolder(String weightsPath) throws FileNotFoundException {
        File weightsFolder = new File(weightsPath);
        if (!weightsFolder.exists()) {
            if (!weightsFolder.mkdirs()) {
                throw new RuntimeException("Can't save weights to output folder");
            }
        }

        layer1.writeWeightsTo(new PrintStream(new FileOutputStream(
                new File(weightsFolder, "layer1.txt")
        )));
        layer4.writeWeightsTo(new PrintStream(new FileOutputStream(
                new File(weightsFolder, "layer4.txt")
        )));
        layer7.writeWeightsTo(new PrintStream(new FileOutputStream(
                new File(weightsFolder, "layer7.txt")
        )));
        layer10.writeWeightsTo(new PrintStream(new FileOutputStream(
                new File(weightsFolder, "layer10.txt")
        )));
    }

    public void forward(double[] input) {
        Util.copy(input, buffer0.values);
        layer1.forward();
        layer2.forward();
        layer3.forward();
        layer4.forward();
        layer5.forward();
        layer6.forward();
        layer7.forward();
        layer8.forward();
        layer9.forward();
        layer10.forward();
        layer11.forward();
    }

    public double backward(double[] expect) {
        double lastError = buffer11.initGradientL2(expect);
        layer11.backward();
        layer10.backward();
        layer9.backward();
        layer8.backward();
        layer7.backward();
        layer6.backward();
        layer5.backward();
        layer4.backward();
        layer3.backward();
        layer2.backward();
        layer1.backward();
        return lastError;
    }

    public void adjust(double learningRate) {
        learningRate /= buffer11.size;
        layer10.adjustWeights(learningRate);
        layer9.adjustWeights(learningRate);
        layer8.adjustWeights(learningRate);
        layer7.adjustWeights(learningRate);
        layer6.adjustWeights(learningRate);
        layer5.adjustWeights(learningRate);
        layer4.adjustWeights(learningRate);
        layer3.adjustWeights(learningRate);
        layer2.adjustWeights(learningRate);
        layer1.adjustWeights(learningRate);
    }


    public double batch(TrainingSet ds, double learningRate) {
        double totalError = 0;
        for (int i = 0; i < ds.N; i++) {
            forward(ds.X[i]);
            totalError += backward(ds.Y[i]);
        }
        // learningRate /= buffer11.size;
        adjust(learningRate);
        if (Double.isNaN(totalError)) {
            throw new RuntimeException("oh, no");
        }
        return totalError / ds.N;
    }

    public double test(TrainingSet ds) {
        int numCorrectAnswers = 0;
        for (int i = 0; i < ds.N; i++) {
            forward(ds.X[i]);

            int maxAnswer = 0;
            double maxConfidence = Double.NEGATIVE_INFINITY;
            for (int j = 0; j < 10; j++) {
                if (buffer11.values[j] > maxConfidence) {
                    maxConfidence = buffer11.values[j];
                    maxAnswer = j;
                }
            }

            if (ds.y[i] == maxAnswer) {
                numCorrectAnswers++;
            }
        }

        return ((double)numCorrectAnswers) / ds.N;
    }
}
