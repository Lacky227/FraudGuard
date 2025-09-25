package org.example;

public class FraudDetectionMain {
    public static void main(String[] args) {
        DataProcessor dataProcessor = new DataProcessor();
        dataProcessor.loadData("/home/artur/IdeaProjects/FraudGuard/FraudGuardModel/src/main/resources/creditcard.csv");
    }
}