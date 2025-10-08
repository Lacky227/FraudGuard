package org.example.utils;

import lombok.experimental.UtilityClass;
import smile.data.DataFrame;
import smile.data.formula.Formula;
import smile.data.vector.IntVector;
import smile.data.vector.ValueVector;

@UtilityClass
public class DataUtils {
    public int[] safeIntLabels(Formula formula, DataFrame data) {
        ValueVector yVec = formula.y(data);
        if (yVec instanceof IntVector) {
            return yVec.toIntArray();
        }
        return yVec.doubleStream().mapToInt(d -> (int) d).toArray();
    }
}
