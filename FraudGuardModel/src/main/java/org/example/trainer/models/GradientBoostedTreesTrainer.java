package org.example.trainer.models;

import org.example.trainer.ModelTrainer;
import smile.classification.Classifier;
import smile.data.DataFrame;
import smile.data.formula.Formula;

public class GradientBoostedTreesTrainer implements ModelTrainer {
    @Override
    public Classifier<double[]> train(DataFrame data, Formula formula) {
        return null;
    }

    @Override
    public Classifier<double[]> hyperparameterSearch(DataFrame data, DataFrame validationData, Formula formula) {
        return ModelTrainer.super.hyperparameterSearch(data, validationData, formula);
    }
}
