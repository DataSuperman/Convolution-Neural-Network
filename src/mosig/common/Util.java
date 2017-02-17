package mosig.common;

import mosig.layers.*;

import java.util.Random;

public final class Util {
    private Util() {
    }

    private static final Random RANDOM = new Random();

    public static void setToZero(double[] buffer) {
        for (int i = 0, n = buffer.length; i < n; i++) {
            buffer[i] = 0;
        }
    }

    public static void setToZero(double[][] buffer) {
        for (double[] row : buffer) {
            setToZero(row);
        }
    }

    public static void setToZero(Image[] out) {
        for (Image image : out) {
            setToZero(image.data);
        }
    }

    public static void setToZero(Kernel[] out) {
        for (Kernel kernel: out) {
            setToZero(kernel.weights);
        }
    }

    public static void setToZero(RgbImage[] out) {
        for (RgbImage image : out) {
            setToZero(image.r.data);
            setToZero(image.g.data);
            setToZero(image.b.data);
        }
    }

    public static void setToZero(RgbKernel[] out) {
        for (RgbKernel kernel : out) {
            setToZero(kernel.r.weights);
            setToZero(kernel.g.weights);
            setToZero(kernel.b.weights);
        }
    }

    public static void setToConstant(double[] buffer, double value) {
        for (int i = 0, n = buffer.length; i < n; i++) {
            buffer[i] = value;
        }
    }

    public static void setToConstant(Image image, double value) {
        setToConstant(image.data, value);
    }

    public static void setToConstant(Image[] images, double value) {
        for (Image image : images) {
            setToConstant(image.data, value);
        }
    }

    public static void initGaussian(double[] out, double mean, double variance) {
        for (int i = 0, n = out.length; i < n; i++) {
            out[i] = RANDOM.nextGaussian() * variance + mean;
        }
    }

    public static void initUniform(double[] out, double min, double max) {
        double scale = max - min;
        double offset = -min;
        for (int i = 0, n = out.length; i < n; i++) {
            out[i] = RANDOM.nextDouble() * scale + offset;
        }
    }

    public static void initWeights(FullyConnectedLayer layer) {
        double max = 1.0 / layer.inputSize;
        for (int i = 0; i < layer.outputSize; i++) {
            initUniform(layer.weights[i], -max, max);
        }
        setToConstant(layer.bias, 0.01);
    }

    public static void initWeights(ActivationLayer layer) {
    }

    public static void initWeights(PoolingLayer layer) {
    }

    public static void initWeights(SoftmaxLayer layer) {
    }

    public static void initWeights(Kernel out) {
        initGaussian(out.weights, 0, 1);
    }

    public static void initWeights(RgbConvolutionLayer layer) {
        for (RgbKernel kernel : layer.kernels) {
            initWeights(kernel.r);
            initWeights(kernel.g);
            initWeights(kernel.b);
        }
        initUniform(layer.biases, 0.01, 0.02);
    }

    public static void initWeights(ConvolutionLayer layer) {
        for (Kernel kernel : layer.kernels) {
            initWeights(kernel);
        }
        initUniform(layer.biases, 0.01, 0.02);
    }

    public static void copy(double[] src, double[] out) {
        System.arraycopy(src, 0, out, 0, Math.max(src.length, out.length));
    }

    public static void copy(double[] src, Image out) {
        copy(src, out.data);
    }

    public static void copy(Image src, double[] out) {
        copy(src.data, out);
    }

    public static void copy(double[] src, RgbImage out) {
        int totalSize = 3 * out.width * out.height;
        if (totalSize != src.length) {
            throw new IllegalArgumentException("Mismatched size");
        }
        int offset = 0;
        System.arraycopy(src, offset, out.r.data, 0, out.r.data.length);
        offset += out.r.data.length;
        System.arraycopy(src, offset, out.g.data, 0, out.g.data.length);
        offset += out.g.data.length;
        System.arraycopy(src, offset, out.b.data, 0, out.b.data.length);
    }

    public static void copy(RgbImage src, double[] out) {
        int totalSize = 3 * src.width * src.height;
        if (totalSize != out.length) {
            throw new IllegalArgumentException("Mismatched size");
        }
        int offset = 0;
        System.arraycopy(src.r.data, 0, out, offset, src.r.data.length);
        offset += src.r.data.length;
        System.arraycopy(src.g.data, 0, out, offset, src.g.data.length);
        offset += src.g.data.length;
        System.arraycopy(src.b.data, 0, out, offset, src.b.data.length);
    }

    public static void copy(double[] src, Image[] out) {
        int totalSize = 0;
        for (Image image : out) {
            totalSize += image.data.length;
        }
        if (totalSize != src.length) {
            throw new IllegalArgumentException("Mismatched size");
        }
        int offset = 0;
        for (Image image : out) {
            System.arraycopy(src, offset, image.data, 0, image.data.length);
            offset += image.data.length;
        }
    }

    public static void copy(Image[] src, double[] out) {
        int totalSize = 0;
        for (Image image : src) {
            totalSize += image.data.length;
        }
        if (totalSize != out.length) {
            throw new IllegalArgumentException("Mismatched size");
        }
        int offset = 0;
        for (Image image : src) {
            System.arraycopy(image.data, 0, out, offset, image.data.length);
            offset += image.data.length;
        }
    }

    public static void copy(double[] src, RgbImage[] out) {
        int totalSize = 0;
        for (RgbImage image : out) {
            totalSize += image.r.data.length;
            totalSize += image.g.data.length;
            totalSize += image.b.data.length;
        }
        if (totalSize != src.length) {
            throw new IllegalArgumentException("Mismatched size");
        }
        int offset = 0;
        for (RgbImage image : out) {
            System.arraycopy(src, offset, image.r.data, 0, image.r.data.length);
            offset += image.r.data.length;
            System.arraycopy(src, offset, image.g.data, 0, image.g.data.length);
            offset += image.g.data.length;
            System.arraycopy(src, offset, image.b.data, 0, image.b.data.length);
            offset += image.b.data.length;
        }
    }

    public static void copy(RgbImage[] src, double[] out) {
        int totalSize = 0;
        for (RgbImage image : src) {
            totalSize += image.r.data.length;
            totalSize += image.g.data.length;
            totalSize += image.b.data.length;
        }
        if (totalSize != out.length) {
            throw new IllegalArgumentException("Mismatched size");
        }
        int offset = 0;
        for (RgbImage image : src) {
            System.arraycopy(image.r.data, 0, out, offset, image.r.data.length);
            offset += image.r.data.length;
            System.arraycopy(image.g.data, 0, out, offset, image.g.data.length);
            offset += image.g.data.length;
            System.arraycopy(image.b.data, 0, out, offset, image.b.data.length);
            offset += image.b.data.length;
        }
    }

    public static double sum(double[] data) {
        double total = 0;
        for (double datum : data) {
            total += datum;
        }
        return total;
    }
}
