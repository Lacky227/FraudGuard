package org.example.processor;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.csv.CSVFormat;
import org.example.utils.SMOTE;
import smile.data.DataFrame;
import smile.data.transform.InvertibleColumnTransform;
import smile.data.type.DataTypes;
import smile.data.type.StructField;
import smile.data.type.StructType;
import smile.data.vector.IntVector;
import smile.feature.transform.Scaler;
import smile.io.Read;
import smile.util.Index;

import java.io.IOException;
import java.net.URISyntaxException;
import java.util.*;

@Slf4j
@Getter
public class DataProcessor {

    private static final double TRAIN_RATIO = 0.8;
    private static final int RANDOM_SEED = 42;

    public record ProcessedData(DataFrame trainData, DataFrame testData, InvertibleColumnTransform scaler) {}

    public ProcessedData loadData(String csvPath) {
        try {
            log.info("Loaded data from file {}", csvPath);

            CSVFormat format = CSVFormat.Builder.create()
                    .setHeader()
                    .setSkipHeaderRecord(true)
                    .build();

            DataFrame data = Read.csv(csvPath, format, getStructType());
            log.warn("Loaded {} rows and {} columns", data.nrow(),  data.ncol());

            analyzeClassDistribution(data);

            DataFrame normalizeData = normalizeFeatures(data);

            var splitDataValues = splitData(normalizeData);

            DataFrame balancedTrainData = balanceClasses(splitDataValues.trainData);

            return new ProcessedData(balancedTrainData, splitDataValues.testData, scaler);
        } catch (IOException | URISyntaxException e) {
            log.error("Error loading data from file {}", csvPath, e);
            throw new RuntimeException(e);
        }
    }

    private void analyzeClassDistribution(DataFrame data){
        log.info("Analyzing class-distribution");
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
        log.info("Successfully: processed data");
    }

    private InvertibleColumnTransform scaler;
    private DataFrame normalizeFeatures(DataFrame data){
        log.info("Normalizing features...");
        String[] featureName = Arrays.stream(data.names())
                .filter(name -> !name.equals("Class"))
                .toArray(String[]::new);

        this.scaler = Scaler.fit(data, featureName);

        log.info("Successfully: normalized features");
        return scaler.apply(data);
    }

    private record SplitData (DataFrame trainData, DataFrame testData){}

    private SplitData splitData(DataFrame data){
        log.info("Splitting into train ({}) / test ({}) ...",
                String.format("%.0f%%", TRAIN_RATIO * 100),
                String.format("%.0f%%", (1 - TRAIN_RATIO) * 100));
        List<Integer> indexes = new ArrayList<>();
        for (int i = 0; i < data.size(); i++) {
            indexes.add(i);
        }
        Collections.shuffle(indexes, new Random(RANDOM_SEED));

        int trainSize = (int) (data.size() * TRAIN_RATIO);

        int[] trainIndexes = indexes.subList(0, trainSize).stream()
                .mapToInt(Integer::intValue)
                .toArray();
        int[] testIndexes = indexes.subList(trainSize, indexes.size()).stream()
                .mapToInt(Integer::intValue)
                .toArray();

        DataFrame trainData = data.get(Index.of(trainIndexes));
        DataFrame testData = data.get(Index.of(testIndexes));

        log.info("Check for class imbalance ...");
        long trainFraud = trainData.stream()
                .filter(row -> row.getDouble("Class") == 1.0).count();
        if (trainFraud == 0){
            log.warn("Warning: train dataset contains no FRAUD cases!");
        }
        log.warn("Fraud count: {}", trainFraud);

        log.info("Successfully split data: train = {} rows, test = {} rows",
                trainData.size(), testData.size());
        return new SplitData(trainData, testData);
    }

    private DataFrame balanceClasses(DataFrame trainData){
        log.info("Balancing classes...");
        try{
            String[] featureName = Arrays.stream(trainData.names())
                    .filter(name -> !name.equals("Class"))
                    .toArray(String[]::new);

            double[][] features = new double[trainData.size()][featureName.length];
            int[] labels = new int[trainData.size()];

            for (int i = 0; i < trainData.size(); i++) {
                for (int j = 0; j < featureName.length; j++) {
                    features[i][j] = trainData.column(featureName[j]).getDouble(i);
                }
                labels[i] = (int) trainData.column("Class").getDouble(i);
            }

            SMOTE<double[]> smote = new SMOTE<>();
            var balancedData = smote.apply(features, labels);

            double[][] balancedFeatures = balancedData.x;
            int[] balancedLabels = balancedData.y;

            var builder = DataFrame.of(balancedFeatures, featureName);

            IntVector classVector = new IntVector("Class", balancedLabels);

            DataFrame balanced = builder.set("Class", classVector);

            log.info("Successfully: balance classes");
            log.warn("After balance classes : rows ({})", balanced.size());
            analyzeClassDistribution(balanced);

            return balanced;
        } catch (Exception e){
            log.error("Error while balance classes for {} rows", trainData.size(), e);
            return trainData;
        }
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
