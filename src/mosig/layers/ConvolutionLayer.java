package mosig.layers;

import mosig.common.*;

import java.io.PrintStream;
import java.util.Scanner;

public class ConvolutionLayer extends Layer {
    public final int inputWidth;
    public final int inputHeight;
    public final int inputDepth;

    public final int kernelWidth;
    public final int kernelHeight;

    public final int outputDepth;

    public final Kernel[] kernels;
    public final double[] biases;

    public final Kernel[] kernelGradients;
    public final double[] biasGradients;

    public NetworkBuffer sharedInput;
    public NetworkBuffer sharedOutput;

    private final Image[] inputBuffer;
    private final Image[] outputBuffer;
    private final Image[] inputGradient;
    private final Image[] outputGradient;

    public ConvolutionLayer(
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

        this.kernels = Kernel.newArray(outputDepth, kernelWidth, kernelHeight);
        this.biases = new double[outputDepth];

        this.kernelGradients = Kernel.newArray(outputDepth, kernelWidth, kernelHeight);
        this.biasGradients = new double[outputDepth];

        this.inputBuffer = Image.newArray(inputDepth, inputWidth, inputHeight);
        this.outputBuffer = Image.newArray(outputDepth, inputWidth, inputHeight);
        this.inputGradient = Image.newArray(inputDepth, inputWidth, inputHeight);
        this.outputGradient = Image.newArray(outputDepth, inputWidth, inputHeight);
    }

    @Override
    public void readWeightsFrom(Scanner input) {
        for (int i = 0; i < outputDepth; i++) {
            Kernel k = kernels[i];

            biases[i] = input.nextDouble();
            for (int y = 0; y < kernelHeight; y++) {
                for (int x = 0; x < kernelWidth; x++) {
                    k.setWeight(x, y, input.nextDouble());
                }
            }
        }
    }

    @Override
    public void writeWeightsTo(PrintStream output) {
        for (int i = 0; i < outputDepth; i++) {
            Kernel k = kernels[i];

            output.print(biases[i]);
            output.print('\n');
            for (int y = 0; y < kernelHeight; y++) {
                for (int x = 0; x < kernelWidth; x++) {
                    output.print(k.getWeight(x, y));
                    output.print(' ');
                }
                output.print('\n');
            }
            output.print('\n');
        }
    }

    public void forward() {
        Util.copy(sharedInput.values, inputBuffer);
        {
            for (int i = 0; i < outputDepth; i++) {
                Kernel k = kernels[i];
                Image outAccumulator = outputBuffer[i];
                Util.setToConstant(outAccumulator, biases[i]);
                for (int j = 0; j < inputDepth; j++) {
                    k.convolution(inputBuffer[j], outAccumulator);
                }
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
                            inputBuffer[j],    // input of backPropConv()
                            kernels[i],        // input of backPropConv()
                            outputGradient[i], // input of backPropConv()
                            inputGradient[j],  // output of backPropConv()
                            kernelGradients[i] // output of backPropConv()
                    );
                }
                biasGradients[i] += Util.sum(outputGradient[i].data);
            }
        }
        Util.copy(inputGradient, sharedInput.gradients);
    }

    public void adjustWeights(double learningRate) {
        for (int i = 0; i < outputDepth; i++) {
            Kernel k = kernels[i];
            Kernel dk = kernelGradients[i];

            for (int y = 0; y < kernelHeight; y++) {
                for (int x = 0; x < kernelWidth; x++) {

                    k.addWeight(x, y, - learningRate * dk.getWeight(x, y));

                }
            }

            biases[i] -= learningRate * biasGradients[i];

        }

        // clear gradients
        Util.setToZero(kernelGradients);
        Util.setToZero(biasGradients);
    }
}
