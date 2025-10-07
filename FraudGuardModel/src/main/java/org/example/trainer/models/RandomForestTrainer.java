package org.example.trainer.models;

import lombok.extern.slf4j.Slf4j;
import org.example.trainer.ModelTrainer;
import smile.classification.Classifier;
import smile.classification.RandomForest;
import smile.data.DataFrame;
import smile.data.Tuple;
import smile.data.formula.Formula;
import smile.data.type.StructType;

@Slf4j
public class RandomForestTrainer implements ModelTrainer {
    private static class RandomForestAdapter implements Classifier<double[]> {
        private final RandomForest model;
        private final Formula formula;
        private final StructType schema;

        RandomForestAdapter(RandomForest model, Formula formula, StructType schema) {
            this.model = model;
            this.formula = formula;
            this.schema = schema;
        }

        @Override
        public int predict(double[] x) {
            Tuple xt = Tuple.of(schema, x);
            return model.predict(formula.x(xt));
        }
        @Override
        public int numClasses() {
            return model.numClasses();
        }

        @Override
        public int[] classes() {
            return model.classes();
        }
    }
    @Override
    public Classifier<double[]> train(DataFrame data, Formula formula) {
        return new RandomForestAdapter(RandomForest.fit(formula, data, new RandomForest.Options(100)),
                formula,
                data.schema());
    }

    @Override
    public Classifier<double[]> hyperparameterSearch(DataFrame trainData, DataFrame validationData, Formula formula) {
        if (trainData == null || validationData == null || formula == null) {
            throw new IllegalArgumentException("Training data, validation data, and formula cannot be null");
        }

        log.info("Starting hyperparameter search for Random Forest...");

        int[] nTrees = {50, 100, 200};
        int[] maxDepths = {10, 20, Integer.MAX_VALUE};
        int[] mtry = {0, (int) Math.sqrt(trainData.ncol()), trainData.ncol() / 3};

        Classifier<double[]> bestModel = null;
        double bestScore = Double.NEGATIVE_INFINITY;

        double[][] validationFeatures = formula.x(validationData).toArray();
        int[] validationLabels = formula.y(validationData).toIntArray();

        StructType schema = trainData.schema();

        for (int trees : nTrees) {
            for (int depth : maxDepths) {
                for (int features : mtry) {
                    try {
                        RandomForest.Options options = new RandomForest.Options(trees, features, depth, 0, 5);
                        Classifier<double[]> model = new RandomForestAdapter(RandomForest.fit(formula, trainData, options), formula, schema);

                        double score = evaluateF1Score(model, validationFeatures, validationLabels);
                        log.info("Trees: {}, MaxDepth: {}, Mtry: {}, F1-Score: {}",
                                trees, depth, features, score);

                        if (score > bestScore) {
                            bestScore = score;
                            bestModel = model;
                        }
                    } catch (IllegalArgumentException e) {
                        log.error("Error for Trees: {}, MaxDepth: {}, Mtry: {}: {}",
                                trees, depth, features, e.getMessage());
                    }
                }
            }
        }

        if (bestModel == null) {
            log.warn("No valid model found. Using default parameters.");
            bestModel = new RandomForestAdapter(RandomForest.fit(formula, trainData, new RandomForest.Options(100, 0, 20, 0, 5)), formula, schema);
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
