package org.example.trainer.models;

import lombok.extern.slf4j.Slf4j;
import org.example.trainer.ModelTrainer;
import smile.classification.Classifier;
import smile.classification.KNN;
import smile.data.DataFrame;
import smile.data.formula.Formula;

@Slf4j
public class KNNTrainer implements ModelTrainer {
    @Override
    public Classifier<double[]> train(DataFrame data, Formula formula) {
        var features = formula.x(data).toArray();
        var labels = formula.y(data).toIntArray();

        return KNN.fit(features, labels, 5);
    }

    @Override
    public Classifier<double[]> hyperparameterSearch(DataFrame trainData, DataFrame validationData, Formula formula) {
        if (trainData == null || validationData == null || formula == null) {
            throw new IllegalArgumentException("Training data, validation data, and formula cannot be null");
        }

        log.info("Starting hyperparameter search for KNN...");

        int[] kValues = {1, 3, 5, 7, 10, 15};

        Classifier<double[]> bestModel = null;
        double bestScore = Double.NEGATIVE_INFINITY;

        var validationFeatures = formula.x(validationData).toArray();
        var validationLabels = formula.y(validationData).toIntArray();

        var trainFeatures = formula.x(trainData).toArray();
        var trainLabels = formula.y(trainData).toIntArray();

        for (int k : kValues) {
            try {
                Classifier<double[]> model = KNN.fit(trainFeatures, trainLabels, k);

                double score = evaluateF1Score(model, validationFeatures, validationLabels);
                log.info("K: {}, F1-Score: {}", k, score);

                if (score > bestScore) {
                    bestScore = score;
                    bestModel = model;
                }
            } catch (IllegalArgumentException e) {
                log.error("Error for K: {}: {}", k, e.getMessage());
            }
        }

        if (bestModel == null) {
            log.warn("No valid model found. Using default parameters.");
            bestModel = KNN.fit(trainFeatures, trainLabels, 5);
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
