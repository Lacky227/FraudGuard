package org.example.trainer.models;

import lombok.extern.slf4j.Slf4j;
import org.example.trainer.ModelTrainer;
import smile.classification.Classifier;
import smile.classification.LogisticRegression;
import smile.data.DataFrame;
import smile.data.formula.Formula;

@Slf4j
public class LogisticRegressionTrainer implements ModelTrainer {

    @Override
    public Classifier<double[]> train(DataFrame data, Formula formula) {
        log.info("Starting training...");
        if (data == null || formula == null) {
            throw new IllegalArgumentException("Data and formula cannot be null");
        }
        double[][] x = formula.x(data).toArray();
        int[] y = formula.y(data).toIntArray();
        LogisticRegression.Options options = new LogisticRegression.Options(0.1, 1.0E-5, 1000);
        return LogisticRegression.fit(x, y, options);
    }

    @Override
    public Classifier<double[]> hyperparameterSearch(DataFrame trainData, DataFrame validationData, Formula formula) {
        if (trainData == null || validationData == null || formula == null) {
            throw new IllegalArgumentException("Training data, validation data, and formula cannot be null");
        }

        log.info("Starting hyperparameter search for Logistic Regression...");

        double[] lambdaValues = {0.001, 0.01, 0.1, 1.0};
        double[] tolerances = {1.0E-6, 1.0E-5, 1.0E-4};
        int[] maxIterations = {500, 1000, 2000};

        Classifier<double[]> bestModel = null;
        double bestScore = Double.NEGATIVE_INFINITY;

        double[][] validationFeatures = formula.x(validationData).toArray();
        int[] validationLabels = formula.y(validationData).toIntArray();

        double[][] trainFeatures = formula.x(trainData).toArray();
        int[] trainLabels = formula.y(trainData).toIntArray();

        for (double lambda : lambdaValues) {
            for (double tol : tolerances) {
                for (int maxIter : maxIterations) {
                    try {
                        LogisticRegression.Options options = new LogisticRegression.Options(lambda, tol, maxIter);
                        Classifier<double[]> model = LogisticRegression.fit(trainFeatures, trainLabels, options);

                        double score = evaluateF1Score(model, validationFeatures, validationLabels);
                        log.info("Lambda: {}, Tolerance: {}, MaxIter: {}, F1-Score: {}",
                                lambda, tol, maxIter, score);

                        if (score > bestScore) {
                            bestScore = score;
                            bestModel = model;
                        }
                    } catch (IllegalArgumentException e) {
                        log.error("Error for Lambda: {}, Tolerance: {}, MaxIter: {}: {}",
                                lambda, tol, maxIter, e.getMessage());
                    }
                }
            }
        }

        if (bestModel == null) {
            log.warn("No valid model found. Using default parameters.");
            bestModel = LogisticRegression.fit(trainFeatures, trainLabels, new LogisticRegression.Options());
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
