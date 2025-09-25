package org.example;

import org.apache.commons.csv.CSVFormat;
import smile.data.DataFrame;
import smile.data.type.DataTypes;
import smile.data.type.StructField;
import smile.data.type.StructType;
import smile.io.Read;

import java.io.IOException;
import java.net.URISyntaxException;

public class DataProcessor {

    public void loadData(String csvPath) {
        try {
            DataFrame data = Read.csv(csvPath, CSVFormat.DEFAULT.withFirstRecordAsHeader(), getStructType());

            analyzeClassDistribution(data);
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
        System.out.printf("Розподіл класів:\n");
        System.out.printf("  Нормальні транзакції: %d (%.2f%%)\n",
                normalCount, (normalCount * 100.0) / data.size());
        System.out.printf("  Шахрайські транзакції: %d (%.2f%%)\n",
                fraudCount, (fraudCount * 100.0) / data.size());
        System.out.printf("  Дисбаланс: 1:%.0f\n", (double) normalCount / fraudCount);
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
