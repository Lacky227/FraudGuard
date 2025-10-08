package org.example.utils;

import lombok.experimental.UtilityClass;
import lombok.extern.slf4j.Slf4j;
import smile.classification.Classifier;

import java.io.*;

@Slf4j
@UtilityClass
public class ModelPersistence {

    public void saveModel(Classifier<double[]> model, String filePath) throws IOException {
        log.info("Saving model...");
        try (ObjectOutputStream oos = new ObjectOutputStream(
                new BufferedOutputStream(new FileOutputStream(filePath)))) {
            oos.writeObject(model);
        }
        log.info("Model successfully saved!");
    }
    public Classifier<double[]> loadModel(String filePath) throws IOException {
        log.info("Loading model...");
        try (ObjectInputStream ois = new ObjectInputStream(
                new BufferedInputStream(new FileInputStream(filePath)))) {
            Classifier<double[]> model = (Classifier<double[]>) ois.readObject();
            log.info("Model successfully loaded!");
            return model;

        } catch (ClassNotFoundException e) {
            throw new RuntimeException(e);
        }
    }
}
