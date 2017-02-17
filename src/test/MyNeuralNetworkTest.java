package test;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintStream;

public class MyNeuralNetworkTest {

    public static void main(String[] args) throws IOException {
        System.out.println("Loading test data. This may take a while");
        TestingSet fullSet = TestingSet.load(
                "src/test/Xte.csv"
        );

        System.out.println("Initializing neural network");
        MyNeuralNetwork nn = new MyNeuralNetwork();

        System.out.println("Loading weights from disk");
        nn.loadFromFolder("weights-16.7000");

        PrintStream out = new PrintStream(new FileOutputStream("src/test/Yte.csv"));
        out.println("Id,Prediction");

        for (int i = 0; i < fullSet.N; i++) {
            nn.forward(fullSet.X[i]);

            double[] Y = nn.buffer11.values;
            int y = 0;
            double max = Y[0];
            for (int j = 1; j < 10; j++) {
                if (max < Y[j]) {
                    max = Y[j];
                    y = j;
                }
            }

            String answer = (i + 1) + "," + y;
            out.println(answer);
            System.out.println(answer);
        }

        out.close();
    }
}
