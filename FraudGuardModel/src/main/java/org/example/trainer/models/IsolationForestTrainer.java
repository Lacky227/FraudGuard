package org.example.trainer.models;

import lombok.extern.slf4j.Slf4j;
import org.example.trainer.ModelTrainer;
import smile.anomaly.IsolationForest;
import smile.classification.Classifier;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.data.type.StructType;

@Slf4j
public class IsolationForestTrainer implements ModelTrainer {

    private static class IsolationForestAdapter implements Classifier<double[]> {
        private final IsolationForest isolationForest;
        private final StructType schema;
        private final double threshold;

        IsolationForestAdapter(IsolationForest isolationForest, StructType schema, double threshold) {
            this.isolationForest = isolationForest;
            this.schema = schema;
            this.threshold = threshold;
        }

        @Override
        public int predict(double[] x) {
            double score = isolationForest.score(x);
            return score > threshold ? 1 : 0;
        }

        @Override
        public boolean soft() {
            return false;
        }

        @Override
        public int numClasses() {
            return 2;
        }

        @Override
        public int[] classes() {
            return new int[]{0, 1};
        }
    }

    @Override
    public Classifier<double[]> train(DataFrame data, Formula formula) {
        if (data == null || formula == null) {
            throw new IllegalArgumentException("Data and formula cannot be null");
        }
        double[][] features = formula.x(data).toArray();
        return new IsolationForestAdapter(
                IsolationForest.fit(features, new IsolationForest.Options(100, 0, 0.7, 0)),
                data.schema(),
                0.5);
    }

    @Override
    public Classifier<double[]> hyperparameterSearch(DataFrame trainData, DataFrame validationData, Formula formula) {
        if (trainData == null || validationData == null || formula == null) {
            throw new IllegalArgumentException("Training data, validation data, and formula cannot be null");
        }

        log.info("Starting hyperparameter search for Isolation Forest...");

        int[] nTrees = {50, 100, 200};
        int[] maxDepths = {0, 10, 20};
        double[] subsamples = {0.5, 0.7, 0.9};
        int[] extensionLevels = {0, 1, 2};
        double[] thresholds = {0.4, 0.5, 0.6};

        Classifier<double[]> bestModel = null;
        double bestScore = Double.NEGATIVE_INFINITY;

        double[][] trainFeatures = formula.x(trainData).toArray();
        double[][] validationFeatures = formula.x(validationData).toArray();
        int[] validationLabels = formula.y(validationData).toIntArray();

        StructType schema = trainData.schema();

        for (int trees : nTrees) {
            for (int maxDepth : maxDepths) {
                for (double subsample : subsamples) {
                    for (int extensionLevel : extensionLevels) {
                        for (double threshold : thresholds) {
                            try {
                                IsolationForest.Options options = new IsolationForest.Options(trees, maxDepth, subsample, extensionLevel);
                                Classifier<double[]> model = new IsolationForestAdapter(
                                        IsolationForest.fit(trainFeatures, options), schema, threshold);

                                double score = evaluateF1Score(model, validationFeatures, validationLabels);
                                log.info("Trees: {}, MaxDepth: {}, Subsample: {}, ExtensionLevel: {}, Threshold: {}, F1-Score: {}",
                                        trees, maxDepth, subsample, extensionLevel, threshold, score);

                                if (score > bestScore) {
                                    bestScore = score;
                                    bestModel = model;
                                }
                            } catch (IllegalArgumentException e) {
                                log.error("Error for Trees: {}, MaxDepth: {}, Subsample: {}, ExtensionLevel: {}, Threshold: {}: {}",
                                        trees, maxDepth, subsample, extensionLevel, threshold, e.getMessage());
                            }
                        }
                    }
                }
            }
        }

        if (bestModel == null) {
            log.warn("No valid model found. Using default parameters.");
            bestModel = new IsolationForestAdapter(
                    IsolationForest.fit(trainFeatures, new IsolationForest.Options(100, 0, 0.7, 0)),
                    schema,
                    0.5);
        }

        log.info("Best F1-Score: {}", bestScore);
        return bestModel;
    }

    private double evaluateF1Score(Classifier<double[]> model, double[][] features, int[] actualLabels) {
        if (model == null || features == null || actualLabels == null) {
            throw new IllegalArgumentException("Model, features, and labels cannot be null");
        }
        if (features.length != actualLabels.length) {
            throw new IllegalArgumentException("Features and labels must have the same length");
        }

        int tp = 0, fp = 0, fn = 0;

        for (int i = 0; i < features.length; i++) {
            int predicted = model.predict(features[i]);
            int actual = actualLabels[i];

            if (predicted == 1 && actual == 1) tp++;
            else if (predicted == 1 && actual == 0) fp++;
            else if (predicted == 0 && actual == 1) fn++;
        }

        double precision = tp + fp > 0 ? (double) tp / (tp + fp) : 0.0;
        double recall = tp + fn > 0 ? (double) tp / (tp + fn) : 0.0;

        return precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0.0;
    }
}
