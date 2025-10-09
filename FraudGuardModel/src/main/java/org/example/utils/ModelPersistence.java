package org.example.utils;

import lombok.experimental.UtilityClass;
import lombok.extern.slf4j.Slf4j;
import org.example.trainer.models.ModelWithScaler;
import smile.classification.Classifier;

import java.io.*;

@Slf4j
@UtilityClass
public class ModelPersistence {

    public void saveModel(ModelWithScaler modelWithScaler, String filePath) throws IOException {
        log.info("Saving model...");
        try (ObjectOutputStream oos = new ObjectOutputStream(
                new BufferedOutputStream(new FileOutputStream(filePath)))) {
            oos.writeObject(modelWithScaler);
        }
        log.info("Model successfully saved!");
    }
    public Classifier<double[]> loadModel(String filePath) throws IOException {
        log.info("Loading model...");
        try (ObjectInputStream ois = new ObjectInputStream(
                new BufferedInputStream(new FileInputStream(filePath)))) {
            ModelWithScaler modelWithScaler = (ModelWithScaler) ois.readObject();
            log.info("Model successfully loaded!");
            return modelWithScaler.getModel();

        } catch (ClassNotFoundException e) {
            throw new RuntimeException(e);
        }
    }
    public ModelWithScaler loadModelWithScaler(String filePath) throws IOException {
        log.info("Loading model...");
        try (ObjectInputStream ois = new ObjectInputStream(
                new BufferedInputStream(new FileInputStream(filePath)))) {
            ModelWithScaler modelWithScaler = (ModelWithScaler) ois.readObject();
            log.info("Model successfully loaded!");
            return modelWithScaler;
        }catch (ClassNotFoundException e) {
            throw new RuntimeException(e);
        }
    }
}
