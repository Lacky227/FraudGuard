package org.example;

import lombok.extern.slf4j.Slf4j;
import org.example.processor.DataProcessor;
import org.example.trainer.ModelTrainer;
import org.example.trainer.evaluator.ModelEvaluator;
import org.example.trainer.models.*;
import smile.data.DataFrame;
import smile.data.formula.Formula;

import java.util.HashMap;
import java.util.Map;

@Slf4j
public class FraudDetectionMain {
    private static final String DATASET_PATH = "/home/artur/IdeaProjects/FraudGuard/FraudGuardModel/src/main/resources/creditcard.csv";

    public static void main(String[] args) {
        log.info("Loading and processing data...");
        DataProcessor dataProcessor = new DataProcessor();
        var processedData = dataProcessor.loadData(DATASET_PATH);

        DataFrame trainData = processedData.trainData();
        DataFrame testData = processedData.testData();

        log.info("Size training data: rows ({})", trainData.size());
        log.info("Size testing data: rows ({})", testData.size());

        log.info("Preparing models...");
        Map<String, ModelTrainer> models = prepareModels();

        log.info("Training and evaluating models...");
        ModelEvaluator evaluator = new ModelEvaluator();
        Map<String, ModelEvaluator.ModelResults> results = new HashMap<>();

        Formula formula = Formula.lhs("Class");

        for (Map.Entry<String, ModelTrainer> entry : models.entrySet()) {
            String modelName = entry.getKey();
            ModelTrainer modelTrainer = entry.getValue();

            log.info("Training model: {}", modelName);
            var model = modelTrainer.train(trainData, formula);

            var modelResult = evaluator.evaluate(model, testData, modelName);
            results.put(modelName, modelResult);

            log.info("Training completed for model: {}", modelName);
        }

        log.info("=".repeat(80));
        log.info("Comparison of model results:");
        evaluator.printComparisonTable(results);

        String bestModelName = evaluator.findBestModel(results);
        var bestModelResult = results.get(bestModelName);

        log.info("=".repeat(80));
        log.info("Best model identified: {}", bestModelName);
        log.info("F1-Score: {}", bestModelResult.f1Score());
        log.info("ROC-AUC: {}", bestModelResult.rocAuc());

        log.info("=".repeat(80));
        log.info("Searching hyperparameters for best model...");
        var optimizedTrainer = models.get(bestModelName);
        var optimizedModel = optimizedTrainer.hyperparameterSearch(trainData, testData, formula);

        var optimizedResult = evaluator.evaluate(optimizedModel, testData, bestModelName + " (Optimized)");

        log.info("Results after optimization:");
        log.info("F1-Score: {} (was {})", optimizedResult.f1Score(), bestModelResult.f1Score());
        log.info("ROC-AUC: {} (was {})", optimizedResult.rocAuc(), bestModelResult.rocAuc());
    }
    private static Map<String, ModelTrainer> prepareModels() {
        Map<String, ModelTrainer> models = new HashMap<>();

        models.put("Logistic Regression", new LogisticRegressionTrainer());
        models.put("Random Forest", new RandomForestTrainer());
        models.put("Gradient Boosted Trees", new GradientBoostedTreesTrainer());
        models.put("K-Nearest Neighbors", new KNNTrainer());
        models.put("Isolation Forest", new IsolationForestTrainer());

        return models;
    }
}