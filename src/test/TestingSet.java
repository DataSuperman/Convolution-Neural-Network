package test;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.util.Scanner;

public class TestingSet {
    public int N;
    public double[][] X;

    private TestingSet(int n, boolean allocateBuffers) {
        N = n;
        if (allocateBuffers) {
            X = new double[n][3072];
        }
    }

    public static TestingSet load(String fileNameX) throws IOException {
        TestingSet ds = new TestingSet(2000, true);
        File cacheX = new File(fileNameX + ".cache");

        if (!cacheX.exists()) {
            ds.parseFromCSV(fileNameX);
        } else {
            ds.loadFromCache(fileNameX);
        }
        return ds;
    }

    private void loadFromCache(String fileNameX) throws IOException {
        byte[] buffer = new byte[64 * 1024];
        File cacheX = new File(fileNameX + ".cache");

        // Load X from .cache file
        {
            ByteBuffer bytesX = ByteBuffer.allocate(2000 * 3072 * 8);
            InputStream inX = new BufferedInputStream(new FileInputStream(cacheX));

            int bytesRead = inX.read(buffer);
            while (bytesRead > 0) {
                bytesX.put(buffer, 0, bytesRead);
                bytesRead = inX.read(buffer);
            }
            inX.close();

            bytesX.rewind();
            DoubleBuffer doublesX = bytesX.asDoubleBuffer();
            for (double[] row : X) {
                doublesX.get(row);
            }
        }

    }

    private static InputStream clearInput(InputStream in) {
        String validChars = "0123456789e-+.";
        return new InputStream() {
            @Override
            public int read() throws IOException {
                int ch = in.read();
                return (validChars.indexOf(ch) != -1) ? ch : ' ';
            }
        };
    }

    private void parseFromCSV(String fileNameX) throws IOException {
        // Parse testing data (slow)
        Scanner inX = new Scanner(new BufferedInputStream(clearInput(new FileInputStream(new File(fileNameX)))));

        for (int i = 0; i < 2000; i++) {
            double[] X_i = X[i];
            for (int j = 0; j < 3072; j++) {
                X_i[j] = inX.nextDouble();
            }
            System.out.println(i + 1);
        }

        // Cache the testing data (so it's faster to load next time)
        {
            // Cache X
            ByteBuffer bytesX = ByteBuffer.allocate(2000 * 3072 * 8);
            DoubleBuffer doublesX = bytesX.asDoubleBuffer();
            for (double[] row : X) {
                doublesX.put(row);
            }
            bytesX.rewind();
            OutputStream outX = new FileOutputStream(new File(fileNameX + ".cache"));
            outX.write(bytesX.array());
            outX.close();
        }

    }

}
