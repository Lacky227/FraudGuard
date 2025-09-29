package org.example;

import smile.classification.Classifier;
import smile.data.DataFrame;
import smile.data.formula.Formula;

public interface ModelTrainer {
    Classifier<double[]> train(DataFrame data, Formula formula);

    default Classifier<double[]> hyperparameterSearch(DataFrame data, DataFrame validationData, Formula formula) {
        return train(data, formula);
    }
}
