package org.example;

import lombok.extern.slf4j.Slf4j;
import org.apache.commons.csv.CSVFormat;
import smile.data.DataFrame;
import smile.data.transform.InvertibleColumnTransform;
import smile.data.type.DataTypes;
import smile.data.type.StructField;
import smile.data.type.StructType;
import smile.feature.transform.Scaler;
import smile.io.Read;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.Arrays;

@Slf4j
public class DataProcessor {

    public void loadData(String csvPath) {
        try {
            log.info("Loading data from file {}", csvPath);
            DataFrame data = Read.csv(csvPath, CSVFormat.DEFAULT.withFirstRecordAsHeader(), getStructType());
            log.warn("Loaded {} rows and {} columns", data.nrow(),  data.ncol());
            log.info("Loaded data from file {}", csvPath);

            analyzeClassDistribution(data);

            DataFrame normalizeData = normalizeFeatures(data);
        } catch (IOException | URISyntaxException e) {
            throw new RuntimeException(e);
        }
    }

    private void analyzeClassDistribution(DataFrame data){
        var classColumn = data.column("Class");
        long fraudCount = 0;
        long normalCount = 0;

        for (int i = 0; i < data.size(); i++) {
            if (classColumn.getDouble(i) == 1.0){
                fraudCount++;
            }else {
                normalCount++;
            }
        }
        log.info("Class distribution:");
        log.info("  Normal transactions: {} ({})",
                normalCount, String.format("%.2f%%", (normalCount * 100.0) / data.size()));
        log.info("  Fraudulent transactions: {} ({})",
                fraudCount, String.format("%.2f%%", (fraudCount * 100.0) / data.size()));
        log.warn("  Imbalance ratio: 1:{}", String.format("%.0f", (double) normalCount / fraudCount));
    }

    private DataFrame normalizeFeatures(DataFrame data){
        log.info("Normalizing features...");
        String[] featureName = Arrays.stream(data.names())
                .filter(name -> !name.equals("Class"))
                .toArray(String[]::new);

        InvertibleColumnTransform scaler = Scaler.fit(data, featureName);

        log.info("Successfully: normalized features");
        return scaler.apply(data);
    }

    private StructType getStructType(){
        return new StructType(
                new StructField("Time", DataTypes.DoubleType),
                new StructField("V1", DataTypes.DoubleType),
                new StructField("V2", DataTypes.DoubleType),
                new StructField("V3", DataTypes.DoubleType),
                new StructField("V4", DataTypes.DoubleType),
                new StructField("V5", DataTypes.DoubleType),
                new StructField("V6", DataTypes.DoubleType),
                new StructField("V7", DataTypes.DoubleType),
                new StructField("V8", DataTypes.DoubleType),
                new StructField("V9", DataTypes.DoubleType),
                new StructField("V10", DataTypes.DoubleType),
                new StructField("V11", DataTypes.DoubleType),
                new StructField("V12", DataTypes.DoubleType),
                new StructField("V13", DataTypes.DoubleType),
                new StructField("V14", DataTypes.DoubleType),
                new StructField("V15", DataTypes.DoubleType),
                new StructField("V16", DataTypes.DoubleType),
                new StructField("V17", DataTypes.DoubleType),
                new StructField("V18", DataTypes.DoubleType),
                new StructField("V19", DataTypes.DoubleType),
                new StructField("V20", DataTypes.DoubleType),
                new StructField("V21", DataTypes.DoubleType),
                new StructField("V22", DataTypes.DoubleType),
                new StructField("V23", DataTypes.DoubleType),
                new StructField("V24", DataTypes.DoubleType),
                new StructField("V25", DataTypes.DoubleType),
                new StructField("V26", DataTypes.DoubleType),
                new StructField("V27", DataTypes.DoubleType),
                new StructField("V28", DataTypes.DoubleType),
                new StructField("Amount", DataTypes.DoubleType),
                new StructField("Class", DataTypes.DoubleType)
        );
    }
}
