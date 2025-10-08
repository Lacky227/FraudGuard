package org.example.trainer.evaluator;

import lombok.extern.slf4j.Slf4j;
import smile.classification.Classifier;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.validation.metric.AUC;
import smile.validation.metric.Accuracy;

import java.util.Map;

@Slf4j
public class ModelEvaluator {
    public record ModelResults(
            String modelName,
            double accuracy,
            double precision,
            double recall,
            double f1Score,
            double rocAuc
    ){}

    public ModelResults evaluate(Classifier<double[]> model, DataFrame testData, String modelName) {
        Formula formula = Formula.lhs("Class");

        var features = formula.x(testData).toArray();
        var actualLabels = formula.y(testData).toIntArray();

        int[] predictions = new int[features.length];
        double[] probabilities = new double[features.length];

        for (int i = 0; i < features.length; i++) {
            predictions[i] = model.predict(features[i]);

            if (model.soft()) {
                double[] posterior = new double[model.numClasses()];
                model.predict(features[i], posterior);
                probabilities[i] = posterior[1];
            } else {
                probabilities[i] = predictions[i];
            }
        }

        double accuracy = Accuracy.of(actualLabels, predictions);

        int tp = 0, fp = 0, fn = 0, tn = 0;
        for (int i = 0; i < actualLabels.length; i++) {
            if (predictions[i] == 1 && actualLabels[i] == 1) tp++;
            else if (predictions[i] == 1 && actualLabels[i] == 0) fp++;
            else if (predictions[i] == 0 && actualLabels[i] == 1) fn++;
            else tn++;
        }

        double precision = tp + fp > 0 ? (double) tp / (tp + fp) : 0.0;
        double recall = tp + fn > 0 ? (double) tp / (tp + fn) : 0.0;
        double f1Score = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0.0;

        double rocAuc = 0.0;
        try {
            rocAuc = AUC.of(actualLabels, probabilities);
        } catch (Exception e) {
            log.warn("Unable to calculate ROC-AUC for {}", modelName);
        }

        log.info("Results for {}:", modelName);
        log.info("  Accuracy:  {}", accuracy);
        log.info("  Precision: {}", precision);
        log.info("  Recall:    {}", recall);
        log.info("  F1-Score:  {}", f1Score);
        log.info("  ROC-AUC:   {}", rocAuc);
        log.info("  Confusion Matrix: TP={}, FP={}, FN={}, TN={}", tp, fp, fn, tn);

        return new ModelResults(modelName, accuracy, precision, recall, f1Score, rocAuc);
    }

    public void printComparisonTable(Map<String, ModelResults> results) {
        log.info(String.format("%-25s %-10s %-10s %-10s %-10s %-10s",
                "Model", "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"));
        log.info("-".repeat(80));

        for (ModelResults result : results.values()) {
            log.info(String.format("%-25s %-10.4f %-10.4f %-10.4f %-10.4f %-10.4f",
                    result.modelName(),
                    result.accuracy(),
                    result.precision(),
                    result.recall(),
                    result.f1Score(),
                    result.rocAuc()));
        }
    }

    public String findBestModel(Map<String, ModelResults> results) {
        String bestModel = "";
        double bestF1Score = 0.0;

        for (Map.Entry<String, ModelResults> entry : results.entrySet()) {
            if (entry.getValue().f1Score() > bestF1Score) {
                bestF1Score = entry.getValue().f1Score();
                bestModel = entry.getKey();
            }
        }

        log.info("Best model based on F1-Score: {} ({})", bestModel, bestF1Score);
        return bestModel;
    }
}
