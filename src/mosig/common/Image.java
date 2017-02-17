package mosig.common;

public final class Image {
    public final int width;
    public final int height;
    public final double[] data;

    public Image(int w, int h) {
        if (w <= 0 || h <= 0) {
            throw new IllegalArgumentException(
                    "all arguments must be strictly positive"
            );
        }
        width = w;
        height = h;
        data = new double[w * h];
    }

    public static Image[] newArray(int n, int w, int h) {
        Image[] arr = new Image[n];
        for (int i = 0; i < n; i++) {
            arr[i] = new Image(w, h);
        }
        return arr;
    }

    public double getPixel(int x, int y) {
        return data[y * width + x];
    }

    public void setPixel(int x, int y, double value) {
        data[y * width + x] = value;
    }

    public void addPixel(int x, int y, double value) {
        data[y * width + x] += value;
    }

    public void getPixels(double[] out, int x, int y, int w, int h) {
        int i = 0;
        for (int row = 0; row < h; ++row) {
            int offset = (y + row) * width + x;
            for (int col = 0; col < w; ++col) {
                out[i++] = data[offset + col];
            }
        }
    }

    public void setToZero() {
        for (int i = 0; i < data.length; i++) {
            data[i] = 0;
        }
    }

    public void setToConstant(double c) {
        for (int i = 0; i < data.length; i++) {
            data[i] = c;
        }
    }
}