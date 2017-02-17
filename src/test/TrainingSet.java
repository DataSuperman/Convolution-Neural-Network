package test;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Scanner;

public class TrainingSet {
    public int N;
    public double[][] X;
    public double[][] Y;
    public int[] y;
    public int[] yHistogram;

    private TrainingSet(int n, boolean allocateBuffers) {
        N = n;
        if (allocateBuffers) {
            X = new double[n][3072];
            Y = new double[n][];
            y = new int[n];
            yHistogram = new int[10];
        }
    }

    public static TrainingSet load(String fileNameX, String fileNameY) throws IOException {
        TrainingSet ds = new TrainingSet(5000, true);
        File cacheX = new File(fileNameX + ".cache");
        File cacheY = new File(fileNameY + ".cache");

        if (!cacheX.exists() || !cacheY.exists()) {
            ds.parseFromCSV(fileNameX, fileNameY);
        } else {
            ds.loadFromCache(fileNameX, fileNameY);
        }
        ds.buildY();
        return ds;
    }

    public void shuffle() {
        ArrayList<Integer> perm = new ArrayList<>(N);
        for (int i = 0; i < N; i++) {
            perm.add(i);
        }
        Collections.shuffle(perm);
        double[][] newX = new double[N][];
        double[][] newY = new double[N][];
        int[] new_y = new int[N];

        for (int i = 0; i < N; i++) {
            int j = perm.get(i);
            newX[i] = X[j];
            newY[i] = Y[j];
            new_y[i] = y[j];
        }
        X = newX;
        Y = newY;
        y = new_y;
    }

    public TrainingSet[] splitByClass() {
        TrainingSet[] result = new TrainingSet[10];
        int[] count = new int[10];
        for (int i = 0; i < 10; i++) {
            int n = yHistogram[i];
            TrainingSet ds = new TrainingSet(n, false);
            ds.X = new double[n][];
            ds.Y = new double[n][];
            ds.y = new int[n];
            ds.yHistogram = new int[10];
            ds.yHistogram[i] = yHistogram[i];
            result[i] = ds;
        }

        for (int i = 0; i < N; i++) {
            TrainingSet ds = result[y[i]];
            int index = count[y[i]]++;
            ds.X[index] = X[i];
            ds.Y[index] = Y[i];
            ds.y[index] = y[i];
        }

        return result;
    }

    public TrainingSet[] splitIntoBatches(int size) {
        assert size > 0;
        int numBatches = N / size;
        TrainingSet[] sets = new TrainingSet[numBatches];

        for (int i = 0; i < numBatches; i++) {
            int first = i * size;
            int last = Math.min(first + size, N);
            sets[i] = subSet(first, last - first);
        }

        return sets;
    }

    public TrainingSet subSet(int first, int count) {
        assert (first < N);
        assert (first + count <= N);
        TrainingSet ds = new TrainingSet(count, false);
        ds.X = new double[count][];
        ds.Y = new double[count][];
        ds.y = new int[count];
        ds.yHistogram = new int[10];

        for (int i = 0; i < count; i++) {
            ds.X[i] = X[first + i];
            ds.Y[i] = Y[first + i];
            ds.y[i] = y[first + i];
            ds.yHistogram[y[first + i]]++;
        }
        return ds;
    }

    private void buildY() {
        double[][] Ys = new double[10][10];
        for (int i = 0; i < 10; i++) {
            Ys[i][i] = 1;
        }
        for (int i = 0; i < 5000; i++) {
            Y[i] = Ys[y[i]];
            yHistogram[y[i]]++;
        }
    }

    private void loadFromCache(String fileNameX, String fileNameY) throws IOException {
        byte[] buffer = new byte[64 * 1024];
        File cacheX = new File(fileNameX + ".cache");
        File cacheY = new File(fileNameY + ".cache");

        // Load X from .cache file
        {
            ByteBuffer bytesX = ByteBuffer.allocate(5000 * 3072 * 8);
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

        // Load Y from .cache file
        {
            ByteBuffer bytesY = ByteBuffer.allocate(5000 * 8);
            InputStream inY = new BufferedInputStream(new FileInputStream(cacheY));

            int bytesRead = inY.read(buffer);
            while (bytesRead > 0) {
                bytesY.put(buffer, 0, bytesRead);
                bytesRead = inY.read(buffer);
            }
            inY.close();

            bytesY.rewind();
            bytesY.asIntBuffer().get(y);
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

    private void parseFromCSV(String fileNameX, String fileNameY) throws IOException {
        // Parse training data (slow)
        Scanner inX = new Scanner(new BufferedInputStream(clearInput(new FileInputStream(new File(fileNameX)))));
        Scanner inY = new Scanner(new BufferedInputStream(clearInput(new FileInputStream(new File(fileNameY)))));

        for (int i = 0; i < 5000; i++) {
            double[] X_i = X[i];
            for (int j = 0; j < 3072; j++) {
                X_i[j] = inX.nextDouble();
            }
            int id = inY.nextInt();
            int output = inY.nextInt();
            y[id - 1] = output;
            System.out.println(i + 1);
        }

        // Cache the training data (so it's faster to load next time)
        {
            // Cache X
            ByteBuffer bytesX = ByteBuffer.allocate(5000 * 3072 * 8);
            DoubleBuffer doublesX = bytesX.asDoubleBuffer();
            for (double[] row : X) {
                doublesX.put(row);
            }
            bytesX.rewind();
            OutputStream outX = new FileOutputStream(new File(fileNameX + ".cache"));
            outX.write(bytesX.array());
            outX.close();
        }

        {
            // Cache Y
            ByteBuffer bytesY = ByteBuffer.allocate(5000 * 8);
            bytesY.asIntBuffer().put(y);
            bytesY.rewind();
            OutputStream outY = new FileOutputStream(new File(fileNameY + ".cache"));
            outY.write(bytesY.array());
            outY.close();
        }
    }

}
