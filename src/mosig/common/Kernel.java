package mosig.common;

public final class Kernel {
    public final int width;
    public final int height;
    public final double[] weights;

    public Kernel(int w, int h) {
        if (w <= 0 || h <= 0) {
            throw new IllegalArgumentException(
                    "all arguments must be strictly positive"
            );
        }
        width = w;
        height = h;
        weights = new double[w * h];
    }

    public static Kernel[] newArray(int n, int w, int h) {
        Kernel[] arr = new Kernel[n];
        for (int i = 0; i < n; i++) {
            arr[i] = new Kernel(w, h);
        }
        return arr;
    }

    public double getWeight(int x, int y) {
        return weights[y * width + x];
    }

    public void setWeight(int x, int y, double value) {
        weights[y * width + x] = value;
    }

    public void addWeight(int x, int y, double value) {
        weights[y * width + x] += value;
    }

    public void convolution(Image src, Image outAccumulator) {
        assert outAccumulator.width == src.width;
        assert outAccumulator.height == src.height;

        int kernCenterOffsetX = width / 2;
        int kernCenterOffsetY = height / 2;

        // for (x, y) in outAccumulator
        for (int outY = 0; outY < outAccumulator.height; outY++) {
            for (int outX = 0; outX < outAccumulator.width; outX++) {


                // for (x, y) in kernel
                for (int kernY = 0; kernY < height; kernY++) {
                    for (int kernX = 0; kernX < width; kernX++) {

                        // source pixel
                        int srcX = outX + kernX - kernCenterOffsetX;
                        int srcY = outY + kernY - kernCenterOffsetY;

                        if (srcX < 0 || srcY < 0 || srcX >= src.width || srcY >= src.height) {
                            // source pixel is outside bounds;
                            // skip it because it's the same as convolution with a 0
                            continue;
                        }

                        // outputPixel += sourcePixel * kernelWeight;
                        double srcPixel = src.getPixel(srcX, srcY);
                        double kernWeight = getWeight(kernX, kernY);
                        outAccumulator.addPixel(outX, outY, srcPixel * kernWeight);

                    }
                }

            }
        }

    }

    public static void backPropConv(
            Image originalInputImage,
            Kernel originalKernel,
            Image outputGradient,
            Image out_inputGradient,
            Kernel out_kernelGradient
    ) {
        int kernCenterOffsetX = originalKernel.width / 2;
        int kernCenterOffsetY = originalKernel.height / 2;

        // For each pixel in `outputGradient`
        for (int gradY = 0; gradY < outputGradient.height; gradY++) {
            for (int gradX = 0; gradX < outputGradient.width; gradX++) {

                double grad = outputGradient.getPixel(gradX, gradY);

                // For each weight in the kernel
                for (int kernY = 0; kernY < originalKernel.height; kernY++) {
                    for (int kernX = 0; kernX < originalKernel.width; kernX++) {
                        int origInX = gradX + kernX - kernCenterOffsetX;
                        int origInY = gradY + kernY - kernCenterOffsetY;

                        if (origInX < 0 || origInY < 0 || origInX >= originalInputImage.width || origInY >= originalInputImage.height) {
                            continue;
                        }

                        double originalWeight = originalKernel.getWeight(kernX, kernY);
                        double originalPixel = originalInputImage.getPixel(origInX, origInY);

                        out_kernelGradient.addWeight(
                                kernX, kernY,
                                grad * originalPixel
                        );

                        out_inputGradient.addPixel(
                                origInX, origInY,
                                grad * originalWeight
                        );

                    }
                }

            }
        }
    }
}
