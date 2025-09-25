package org.example;

import smile.data.DataFrame;
import smile.io.Read;

import java.io.IOException;
import java.net.URISyntaxException;

public class DataProcessor {
    public void loadData(String csvPath) {
        try {
            DataFrame data = Read.csv(csvPath);
        } catch (IOException | URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }
}
