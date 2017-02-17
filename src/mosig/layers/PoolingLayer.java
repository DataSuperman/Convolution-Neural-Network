package mosig.layers;

import mosig.common.Image;
import mosig.common.Layer;
import mosig.common.NetworkBuffer;
import mosig.common.Util;

public final class PoolingLayer extends Layer {
    public final int inputDepth;
    public final int inputWidth;
    public final int inputHeight;

    public final int bucketWidth;
    public final int bucketHeight;

    public final int outputWidth;
    public final int outputHeight;

    public NetworkBuffer sharedInput;
    public NetworkBuffer sharedOutput;

    private final int[][] inputChoices;
    private final Image[] inputBuffer;
    private final Image[] outputBuffer;
    private final Image[] inputGradient;
    private final Image[] outputGradient;

    public PoolingLayer(
            int inputWidth,
            int inputHeight,
            int inputDepth,
            int bucketWidth,
            int bucketHeight
    ) {
        this.inputDepth = inputDepth;
        this.inputWidth = inputWidth;
        this.inputHeight = inputHeight;
        this.bucketWidth = bucketWidth;
        this.bucketHeight = bucketHeight;

        outputWidth = inputWidth / bucketWidth;
        outputHeight = inputHeight / bucketHeight;

        inputChoices = new int[inputDepth][outputWidth * outputHeight];
        inputBuffer = Image.newArray(inputDepth, inputWidth, inputHeight);
        outputBuffer = Image.newArray(inputDepth, outputWidth, outputHeight);
        inputGradient = Image.newArray(inputDepth, inputWidth, inputHeight);
        outputGradient = Image.newArray(inputDepth, outputWidth, outputHeight);
    }

    public void forward() {

        Util.copy(sharedInput.values, inputBuffer);
        {
            // For each image plane
            for (int i = 0; i < inputDepth; i++) {
                int[] choices = inputChoices[i];
                Image inputImage = inputBuffer[i];
                Image outputImage = outputBuffer[i];

                // For each output pixel (x, y)
                for (int y = 0; y < outputHeight; y++) {
                    for (int x = 0; x < outputWidth; x++) {

                        double max = Double.NEGATIVE_INFINITY;
                        int choice = 0;

                        // For each input pixel (x..(x+bucketWidth), y..(y+bucketHeight))
                        for (int poolY = 0; poolY < bucketHeight; ++poolY) {
                            for (int poolX = 0; poolX < bucketWidth; poolX++) {

                                double pixel = inputImage.getPixel(x + poolX, y + poolY);
                                if (pixel > max) {
                                    // Keep the maximum
                                    max = pixel;

                                    // Pack the (poolX, poolY) coordinates into a single int
                                    choice = poolY * bucketWidth + poolX;
                                }

                            }
                        }

                        // set output
                        choices[outputWidth * y + x] = choice;
                        outputImage.setPixel(x, y, max);

                    }
                }

            }
        }
        Util.copy(outputBuffer, sharedOutput.values);
    }

    public void backward() {
        Util.copy(sharedOutput.gradients, outputGradient);
        {
            Util.setToZero(inputGradient);
            for (int i = 0; i < inputDepth; i++) {
                Image outputImageGradient = outputGradient[i];
                Image inputImageGradient = inputGradient[i];
                int[] choices = inputChoices[i];

                for (int y = 0; y < outputHeight; y++) {
                    for (int x = 0; x < outputWidth; x++) {

                        int choice = choices[y * outputHeight + x];
                        int choiceX = choice % bucketWidth;
                        int choiceY = choice / bucketWidth;
                        inputImageGradient.setPixel(
                                choiceX + x,
                                choiceY + y,
                                outputImageGradient.getPixel(x, y)
                        );

                    }
                }

            }
        }
        Util.copy(inputGradient, sharedInput.gradients);
    }

    public void adjustWeights(double learningRate) {
    }

}
