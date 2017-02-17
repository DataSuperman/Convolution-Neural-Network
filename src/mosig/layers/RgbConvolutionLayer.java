package mosig.layers;

import mosig.common.*;

import java.io.PrintStream;
import java.util.Scanner;

public class RgbConvolutionLayer extends Layer {
    public final int inputWidth;
    public final int inputHeight;
    public final int inputDepth;
    public final int kernelWidth;
    public final int kernelHeight;
    public final int outputDepth;

    public final RgbKernel[] kernels;
    public final double[] biases;

    public final RgbKernel[] kernelGradients;
    public final double[] biasGradients;

    public NetworkBuffer sharedInput;
    public NetworkBuffer sharedOutput;

    private final RgbImage[] inputBuffer;
    private final Image[] outputBuffer;
    private final RgbImage[] inputGradient;
    private final Image[] outputGradient;

    public RgbConvolutionLayer(
            int inputWidth,
            int inputHeight,
            int inputDepth,
            int kernelWidth,
            int kernelHeight,
            int outputDepth
    ) {
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.inputDepth = inputDepth;
        this.kernelWidth = kernelWidth;
        this.kernelHeight = kernelHeight;
        this.outputDepth = outputDepth;

        this.kernels = RgbKernel.newArray(outputDepth, kernelWidth, kernelHeight);
        this.biases = new double[outputDepth];

        this.kernelGradients = RgbKernel.newArray(outputDepth, kernelWidth, kernelHeight);
        this.biasGradients = new double[outputDepth];

        this.inputBuffer = RgbImage.newArray(inputDepth, inputWidth, inputHeight);
        this.outputBuffer = Image.newArray(outputDepth, inputWidth, inputHeight);
        this.inputGradient = RgbImage.newArray(inputDepth, inputWidth, inputHeight);
        this.outputGradient = Image.newArray(outputDepth, inputWidth, inputHeight);
    }

    @Override
    public void readWeightsFrom(Scanner input) {
        for (int i = 0; i < outputDepth; i++) {
            RgbKernel k = kernels[i];

            biases[i] = input.nextDouble();
            for (int y = 0; y < kernelHeight; y++) {
                for (int x = 0; x < kernelWidth; x++) {
                    k.r.setWeight(x, y, input.nextDouble());
                }
            }
            for (int y = 0; y < kernelHeight; y++) {
                for (int x = 0; x < kernelWidth; x++) {
                    k.g.setWeight(x, y, input.nextDouble());
                }
            }
            for (int y = 0; y < kernelHeight; y++) {
                for (int x = 0; x < kernelWidth; x++) {
                    k.b.setWeight(x, y, input.nextDouble());
                }
            }
        }
    }

    @Override
    public void writeWeightsTo(PrintStream output) {
        for (int i = 0; i < outputDepth; i++) {
            RgbKernel k = kernels[i];

            output.print(biases[i]);
            output.print("\n\n");
            for (int y = 0; y < kernelHeight; y++) {
                for (int x = 0; x < kernelWidth; x++) {
                    output.print(k.r.getWeight(x, y));
                    output.print(' ');
                }
                output.print('\n');
            }
            output.print('\n');
            for (int y = 0; y < kernelHeight; y++) {
                for (int x = 0; x < kernelWidth; x++) {
                    output.print(k.g.getWeight(x, y));
                    output.print(' ');
                }
                output.print('\n');
            }
            output.print('\n');
            for (int y = 0; y < kernelHeight; y++) {
                for (int x = 0; x < kernelWidth; x++) {
                    output.print(k.b.getWeight(x, y));
                    output.print(' ');
                }
                output.print('\n');
            }
            output.print("\n\n");
        }
    }

    public void forward() {
        Util.copy(sharedInput.values, inputBuffer);

        for (int i = 0; i < outputDepth; i++) {
            RgbKernel k = kernels[i];
            Image outAccumulator = outputBuffer[i];
            Util.setToConstant(outAccumulator, biases[i]);
            for (int j = 0; j < inputDepth; j++) {
                k.convolution(inputBuffer[j], outAccumulator);
            }
        }

        Util.copy(outputBuffer, sharedOutput.values);
    }

    public void backward() {
        Util.copy(sharedOutput.gradients, outputGradient);
        {
            Util.setToZero(inputGradient);

            for (int i = 0; i < outputDepth; i++) {
                for (int j = 0; j < inputDepth; j++) {
                    Kernel.backPropConv(
                            inputBuffer[j].r,  // input of backPropConv()
                            kernels[i].r,      // input of backPropConv()
                            outputGradient[i], // input of backPropConv()
                            inputGradient[j].r,// output of backPropConv()
                            kernelGradients[i].r// output of backPropConv()
                    );
                    Kernel.backPropConv(
                            inputBuffer[j].g,  // input of backPropConv()
                            kernels[i].g,      // input of backPropConv()
                            outputGradient[i], // input of backPropConv()
                            inputGradient[j].g,// output of backPropConv()
                            kernelGradients[i].g// output of backPropConv()
                    );
                    Kernel.backPropConv(
                            inputBuffer[j].b,  // input of backPropConv()
                            kernels[i].b,      // input of backPropConv()
                            outputGradient[i], // input of backPropConv()
                            inputGradient[j].b,// output of backPropConv()
                            kernelGradients[i].b// output of backPropConv()
                    );
                }
                biasGradients[i] += Util.sum(outputGradient[i].data);
            }
        }
        Util.copy(inputGradient, sharedInput.gradients);
    }

    public void adjustWeights(double learningRate) {
        for (int i = 0; i < outputDepth; i++) {
            RgbKernel k = kernels[i];
            RgbKernel dk = kernelGradients[i];

            for (int y = 0; y < kernelHeight; y++) {
                for (int x = 0; x < kernelWidth; x++) {

                    k.r.addWeight(x, y, - learningRate * dk.r.getWeight(x, y));
                    k.g.addWeight(x, y, - learningRate * dk.g.getWeight(x, y));
                    k.b.addWeight(x, y, - learningRate * dk.b.getWeight(x, y));

                }
            }

            biases[i] -= learningRate * biasGradients[i];

        }

        // Clear weight gradients
        Util.setToZero(kernelGradients);
        Util.setToZero(biasGradients);
    }
}
