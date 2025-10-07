package org.example.trainer.models;

import lombok.extern.slf4j.Slf4j;
import org.example.trainer.ModelTrainer;
import smile.classification.Classifier;
import smile.classification.GradientTreeBoost;
import smile.data.DataFrame;
import smile.data.Tuple;
import smile.data.formula.Formula;
import smile.data.type.StructType;

@Slf4j
public class GradientBoostedTreesTrainer implements ModelTrainer {

    private static class GradientBoostedTreesAdapter implements Classifier<double[]> {
        private final GradientTreeBoost gtb;
        private final Formula formula;
        private final StructType schema;

        GradientBoostedTreesAdapter(GradientTreeBoost gtb, Formula formula, StructType schema) {
            this.gtb = gtb;
            this.formula = formula;
            this.schema = schema;
        }

        @Override
        public int predict(double[] x) {
            Tuple xt = Tuple.of(schema, x);
            return gtb.predict(formula.x(xt));
        }

        @Override
        public int numClasses() {
            return gtb.numClasses();
        }

        @Override
        public int[] classes() {
            return gtb.classes();
        }
    }

    @Override
    public Classifier<double[]> train(DataFrame data, Formula formula) {
        if (data == null || formula == null) {
            throw new IllegalArgumentException("Data and formula cannot be null");
        }
        return new GradientBoostedTreesAdapter(
                GradientTreeBoost.fit(formula, data, new GradientTreeBoost.Options(100)),
                formula,
                data.schema());
    }

    @Override
    public Classifier<double[]> hyperparameterSearch(DataFrame trainData, DataFrame validationData, Formula formula) {
        if (trainData == null || validationData == null || formula == null) {
            throw new IllegalArgumentException("Training data, validation data, and formula cannot be null");
        }

        log.info("Starting hyperparameter search for Gradient Boosted Trees...");

        int[] nTrees = {50, 100, 200};
        int[] maxDepths = {3, 5, 7};
        int[] nodeSizes = {5, 10, 20};
        double[] shrinkages = {0.01, 0.05, 0.1};
        double[] subsamples = {0.5, 0.7, 1.0};

        Classifier<double[]> bestModel = null;
        double bestScore = Double.NEGATIVE_INFINITY;

        double[][] validationFeatures = formula.x(validationData).toArray();
        int[] validationLabels = formula.y(validationData).toIntArray();

        StructType schema = trainData.schema();

        for (int trees : nTrees) {
            for (int depth : maxDepths) {
                for (int nodeSize : nodeSizes) {
                    for (double shrinkage : shrinkages) {
                        for (double subsample : subsamples) {
                            try {
                                GradientTreeBoost.Options options = new GradientTreeBoost.Options(trees, depth, 6, nodeSize, shrinkage, subsample, null, null);
                                Classifier<double[]> model = new GradientBoostedTreesAdapter(
                                        GradientTreeBoost.fit(formula, trainData, options), formula, schema);

                                double score = evaluateF1Score(model, validationFeatures, validationLabels);
                                log.info("Trees: {}, MaxDepth: {}, NodeSize: {}, Shrinkage: {}, Subsample: {}, F1-Score: {:.4f}",
                                        trees, depth, nodeSize, shrinkage, subsample, score);

                                if (score > bestScore) {
                                    bestScore = score;
                                    bestModel = model;
                                }
                            } catch (IllegalArgumentException e) {
                                log.error("Error for Trees: {}, MaxDepth: {}, NodeSize: {}, Shrinkage: {}, Subsample: {}: {}",
                                        trees, depth, nodeSize, shrinkage, subsample, e.getMessage());
                            }
                        }
                    }
                }
            }
        }

        if (bestModel == null) {
            log.warn("No valid model found. Using default parameters.");
            bestModel = new GradientBoostedTreesAdapter(
                    GradientTreeBoost.fit(formula, trainData, new GradientTreeBoost.Options(100, 20, 6, 5, 0.05, 0.7, null, null)),
                    formula, schema);
        }

        log.info("Best F1-Score: {:.4f}", bestScore);
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
