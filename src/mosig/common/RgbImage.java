package mosig.common;

public class RgbImage {
    public final int width, height;
    public final Image r, g, b;

    public RgbImage(int w, int h) {
        width = w;
        height = h;
        r = new Image(w, h);
        g = new Image(w, h);
        b = new Image(w, h);
    }

    public static RgbImage[] newArray(int count, int w, int h) {
        RgbImage[] array = new RgbImage[count];
        for (int i = 0, n = array.length; i < n; i++) {
            array[i] = new RgbImage(w, h);
        }
        return array;
    }

    public void copy(RgbImage src) {
        assert src.width == width;
        assert src.height == height;

        int rawSize = width * height;
        System.arraycopy(r.data, 0, src.r.data, 0, rawSize);
        System.arraycopy(g.data, 0, src.g.data, 0, rawSize);
        System.arraycopy(b.data, 0, src.b.data, 0, rawSize);
    }

    public void getRaw(double[] out, int outOffset) {
        int rawSize = width * height;
        System.arraycopy(r.data, 0, out, outOffset, rawSize);
        outOffset += rawSize;
        System.arraycopy(g.data, 0, out, outOffset, rawSize);
        outOffset += rawSize;
        System.arraycopy(b.data, 0, out, outOffset, rawSize);
    }

    public void setRaw(double[] src, int srcOffset) {
        int rawSize = width * height;
        System.arraycopy(src, srcOffset, r.data, 0, rawSize);
        srcOffset += rawSize;
        System.arraycopy(src, srcOffset, g.data, 0, rawSize);
        srcOffset += rawSize;
        System.arraycopy(src, srcOffset, b.data, 0, rawSize);
    }

    public void getRawInterleaved(double[] out, int outOffset) {
        int srcOffset = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                out[outOffset++] = r.data[srcOffset];
                out[outOffset++] = g.data[srcOffset];
                out[outOffset++] = b.data[srcOffset++];
            }
        }
    }

    public void setRawInterleaved(double[] src, int srcOffset) {
        int outOffset = 0;
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                r.data[outOffset] = src[srcOffset++];
                g.data[outOffset] = src[srcOffset++];
                b.data[outOffset++] = src[srcOffset++];
            }
        }
    }

}
