package test;

import java.io.IOException;

public class MyNeuralNetworkTrain {

    public static void main(String[] args) throws IOException {
        System.out.println("Loading data set. This may take a while");
        TrainingSet fullSet = TrainingSet.load(
                "src/test/Xtr.csv",
                "src/test/Ytr.csv"
        );

        System.out.println("Splitting data set into batches");
        TrainingSet trainingDataSet = fullSet.subSet(0, 4000);
        TrainingSet testingDataSet = fullSet.subSet(4000, 1000);
        TrainingSet[] trainingBatches = trainingDataSet.splitIntoBatches(4);

        System.out.println("Initializing neural network");
        MyNeuralNetwork nn = new MyNeuralNetwork();

        System.out.println("Loading weights from disk");
        nn.loadFromFolder("weights-17.5000");

        double learningRateMinimum = 0.0001;
        double learningRateMaximum = 0.003;
        double learningRate = 0.1;
        double lastError = 0;

        int globalIteration = 0;
        int saveWeightsEvery = 50;

        long epochStart = System.currentTimeMillis();
        int epoch = 0;
        int i = 0;

        System.out.println();
        while (true) {
            TrainingSet trainingBatch = trainingBatches[i];

            double trainingError = nn.batch(trainingBatch, learningRate);
            System.out.println(String.format(
                    "Epoch %02d, batch %03d ||| learning rate %1.13f, training error %3.13f.",
                    epoch, i, learningRate, trainingError
            ));

/*
            if (trainingError > lastError) {
                learningRate = Math.max(learningRateMinimum, learningRate * 0.9);
            } else {
                learningRate = Math.min(learningRateMaximum, learningRate * 1.1);
            } */
            lastError = trainingError;

            i++;
            if (i >= trainingBatches.length) {
                epoch++;
                learningRate = learningRate*Math.exp(-0.01*epoch);
                i = 0;

                System.out.println("===================================");
                // Display elapsed time
                {
                    long timeSpentMs = System.currentTimeMillis() - epochStart;
                    long timeSpentSec = timeSpentMs / 1000;
                    long timeSpentMin = timeSpentSec / 60;
                    long timeSpentHrs = timeSpentMin / 60;
                    timeSpentSec %= 60;
                    timeSpentMs %= 1000;
                    timeSpentMin %= 60;
                    System.out.println(String.format(
                            "Time spent: %02d:%02d:%02d.%03d",
                            timeSpentHrs, timeSpentMin, timeSpentSec, timeSpentMs
                    ));
                }

                // Display some statistics
                {
                    //double trainingAccuracy = nn.test(trainingDataSet);
                    double testingAccuracy = nn.test(testingDataSet);
                    //double totalAccuracy = (4 * trainingAccuracy + testingAccuracy) / 5;
                    System.out.println(String.format("ACCURACY OVER TESTING-ONLY DATA SET: %03.1f%%", testingAccuracy * 100));
                    //System.out.println(String.format("ACCURACY OVER ENTIRE DATA SET:       %03.1f%%", totalAccuracy * 100));
                    nn.saveToFolder(String.format("weights-%.4f", testingAccuracy * 100));
                }

                // Shuffle data set before start of new epoch
                {
                    trainingDataSet.shuffle();
                    trainingBatches = trainingDataSet.splitIntoBatches(4);
                }
                System.out.println("===================================");
                epochStart = System.currentTimeMillis();
            }

            globalIteration++;
            if (globalIteration % saveWeightsEvery == 0) {
                System.out.println("SAVING");
                nn.saveToFolder("weights");
            }
        }

    }

}
