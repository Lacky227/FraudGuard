package org.example.trainer.models;

import lombok.AllArgsConstructor;
import lombok.Getter;
import smile.classification.Classifier;
import smile.data.transform.InvertibleColumnTransform;

import java.io.Serializable;

@Getter
@AllArgsConstructor
public class ModelWithScaler implements Serializable {
    private static final long serialVersionUID = 1L;
    private final Classifier<double[]> model;
    private final InvertibleColumnTransform scaler;
}
