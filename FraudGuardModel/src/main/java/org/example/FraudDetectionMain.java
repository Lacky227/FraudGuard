package org.example;

import lombok.extern.slf4j.Slf4j;
import org.example.processor.DataProcessor;
import org.example.trainer.ModelTrainer;
import org.example.trainer.evaluator.ModelEvaluator;
import org.example.trainer.models.*;
import org.example.utils.ModelPersistence;
import smile.data.DataFrame;
import smile.data.formula.Formula;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

@Slf4j
public class FraudDetectionMain {
    private static final String DATASET_PATH = "/home/artur/IdeaProjects/FraudGuard/FraudGuardModel/src/main/resources/creditcard.csv";
    private static final String MODEL_SAVE_PATH = "/home/artur/IdeaProjects/FraudGuard/FraudGuardModel/best_fraud_model.bin";

    public static void main(String[] args) {
        log.info("Loading and processing data...");
        DataProcessor dataProcessor = new DataProcessor();
        var processedData = dataProcessor.loadData(DATASET_PATH);

        DataFrame trainData = processedData.trainData();
        DataFrame testData = processedData.testData();

        log.info("Training data size: {} rows", trainData.size());
        log.info("Testing data size: {} rows", testData.size());

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
        log.info("=".repeat(80));

        try {
            log.info("Saving model...");
            ModelPersistence.saveModel(optimizedModel, MODEL_SAVE_PATH);
            log.info("Model saved successfully: {}", MODEL_SAVE_PATH);

            log.info("Verifying model loading...");
            var loadedModel = ModelPersistence.loadModel(MODEL_SAVE_PATH);
            var loadedResult = evaluator.evaluate(loadedModel, testData, bestModelName + " (Loaded)");
            log.info("Loaded model F1-Score: {} (was {})", loadedResult.f1Score(), bestModelResult.f1Score());
        } catch (IOException e) {
            log.error("Failed to save model", e);
            throw new RuntimeException(e);
        }

        log.info("=".repeat(80));
        System.out.println("FINAL REPORT");
        System.out.println("=".repeat(80));
        System.out.println("Best model: " + bestModelName);
        System.out.printf("Accuracy: %.4f%n", optimizedResult.accuracy());
        System.out.printf("Precision: %.4f%n", optimizedResult.precision());
        System.out.printf("Recall: %.4f%n", optimizedResult.recall());
        System.out.printf("F1-Score: %.4f%n", optimizedResult.f1Score());
        System.out.printf("ROC-AUC: %.4f%n", optimizedResult.rocAuc());

        System.out.println("\nRecommendations:");
        if (optimizedResult.f1Score() > 0.85) {
            System.out.println("The model shows excellent performance.");
        } else if (optimizedResult.f1Score() > 0.75) {
            System.out.println("The model shows good performance.");
        } else {
            System.out.println("! The model requires further tuning.");
        }

        if (optimizedResult.recall() < 0.8) {
            System.out.println("! Low recall — some fraudulent transactions may be missed.");
        }

        if (optimizedResult.precision() < 0.8) {
            System.out.println("! Low precision — too many legitimate transactions may be flagged as fraud.");
        }
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