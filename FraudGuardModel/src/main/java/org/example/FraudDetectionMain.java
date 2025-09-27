package org.example;

import lombok.extern.slf4j.Slf4j;
import smile.data.DataFrame;

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
    }
}