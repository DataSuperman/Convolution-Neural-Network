package mosig.common;

public final class RgbKernel {
    public final Kernel r, g, b;
    public final int width, height;

    public RgbKernel(int w, int h) {
        r = new Kernel(w, h);
        g = new Kernel(w, h);
        b = new Kernel(w, h);
        width = w;
        height = h;
    }

    public static RgbKernel[] newArray(int n, int w, int h) {
        RgbKernel[] array = new RgbKernel[n];
        for (int i = 0; i < n; i++) {
            array[i] = new RgbKernel(w, h);
        }
        return array;
    }

    public void convolution(RgbImage src, Image outAccumulator) {
        assert outAccumulator.width == src.width;
        assert outAccumulator.height == src.height;

        r.convolution(src.r, outAccumulator);
        g.convolution(src.g, outAccumulator);
        b.convolution(src.b, outAccumulator);
    }

}
