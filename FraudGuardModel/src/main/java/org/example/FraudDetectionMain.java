package org.example;

import lombok.extern.slf4j.Slf4j;

@Slf4j
public class FraudDetectionMain {
    private static final String DATASET_PATH = "/home/artur/IdeaProjects/FraudGuard/FraudGuardModel/src/main/resources/creditcard.csv";

    public static void main(String[] args) {
        log.info("Loading and processing data...");
        DataProcessor dataProcessor = new DataProcessor();
        dataProcessor.loadData(DATASET_PATH);
    }
}