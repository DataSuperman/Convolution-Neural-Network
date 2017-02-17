package mosig.common;

import java.io.PrintStream;
import java.util.Scanner;

public abstract class Layer {

    public void forward() {
        throw new RuntimeException("not implemented");
    }

    public void backward() {
        throw new RuntimeException("not implemented");
    }

    public void adjustWeights(double learningRate) {
        throw new RuntimeException("not implemented");
    }

    public void readWeightsFrom(Scanner input) {
        throw new RuntimeException("not implemented");
    }

    public void writeWeightsTo(PrintStream output) {
        throw new RuntimeException("not implemented");
    }

}
