package org.example.utils;

import smile.neighbor.KDTree;
import java.util.*;
import java.util.stream.Collectors;

/**
 * SMOTE (Synthetic Minority Oversampling Technique) implementation
 * using Smile KDTree for efficient nearest neighbor search
 */
public class SMOTE<T> {

    private int k = 5; // number of nearest neighbors
    private Random random = new Random();

    /**
     * Result of SMOTE application
     */
    public static class SMOTEResult {
        public final double[][] x;
        public final int[] y;

        public SMOTEResult(double[][] x, int[] y) {
            this.x = x;
            this.y = y;
        }
    }

    /**
     * Default constructor
     */
    public SMOTE() {
        this(5);
    }

    /**
     * Constructor with specified number of neighbors
     * @param k number of nearest neighbors
     */
    public SMOTE(int k) {
        this.k = k;
    }

    /**
     * Constructor with full configuration
     * @param k number of nearest neighbors
     * @param seed seed for random number generator
     */
    public SMOTE(int k, long seed) {
        this.k = k;
        this.random = new Random(seed);
    }

    /**
     * Apply SMOTE for class balancing
     * @param features feature matrix [n_samples][n_features]
     * @param labels array of class labels
     * @return balanced dataset
     */
    public SMOTEResult apply(double[][] features, int[] labels) {
        if (features.length != labels.length) {
            throw new IllegalArgumentException("Number of samples and labels must match");
        }

        // Analyze class distribution
        Map<Integer, List<Integer>> classIndices = analyzeClassDistribution(labels);

        if (classIndices.size() < 2) {
            throw new IllegalArgumentException("At least 2 classes required");
        }

        // Find majority class size
        int maxClassSize = classIndices.values().stream()
                .mapToInt(List::size)
                .max()
                .orElse(0);

        System.out.printf("Class distribution before SMOTE:\n");
        classIndices.forEach((classLabel, indices) ->
                System.out.printf("  Class %d: %d samples\n", classLabel, indices.size()));

        // Create balanced dataset
        List<double[]> balancedFeatures = new ArrayList<>(Arrays.asList(features));
        List<Integer> balancedLabels = new ArrayList<>();
        for (int label : labels) {
            balancedLabels.add(label);
        }

        // Generate synthetic samples for minority classes
        for (Map.Entry<Integer, List<Integer>> entry : classIndices.entrySet()) {
            int classLabel = entry.getKey();
            List<Integer> indices = entry.getValue();

            if (indices.size() < maxClassSize) {
                int samplesToGenerate = maxClassSize - indices.size();
                System.out.printf("Generating %d synthetic samples for class %d\n",
                        samplesToGenerate, classLabel);

                List<double[]> syntheticSamples = generateSyntheticSamples(
                        features, indices, samplesToGenerate);

                balancedFeatures.addAll(syntheticSamples);
                for (int i = 0; i < samplesToGenerate; i++) {
                    balancedLabels.add(classLabel);
                }
            }
        }

        // Convert result
        double[][] resultFeatures = balancedFeatures.toArray(new double[0][]);
        int[] resultLabels = balancedLabels.stream().mapToInt(i -> i).toArray();

        System.out.printf("Class distribution after SMOTE:\n");
        Map<Integer, List<Integer>> balancedClassIndices = analyzeClassDistribution(resultLabels);
        balancedClassIndices.forEach((classLabel, indices) ->
                System.out.printf("  Class %d: %d samples\n", classLabel, indices.size()));

        return new SMOTEResult(resultFeatures, resultLabels);
    }

    /**
     * Generate synthetic samples for minority class
     * @param features original feature matrix
     * @param classIndices indices of samples belonging to the class
     * @param samplesToGenerate number of synthetic samples to generate
     * @return list of synthetic samples
     */
    private List<double[]> generateSyntheticSamples(double[][] features,
                                                    List<Integer> classIndices,
                                                    int samplesToGenerate) {
        List<double[]> syntheticSamples = new ArrayList<>();

        if (classIndices.size() < k) {
            // If fewer samples than k, use all available
            System.out.printf("Warning: only %d samples available for class, " +
                            "which is less than k=%d. Using all available.\n",
                    classIndices.size(), k);
        }

        // Prepare data for KDTree
        double[][] classFeatures = classIndices.stream()
                .map(i -> features[i])
                .toArray(double[][]::new);

        // Create KDTree for efficient neighbor search
        KDTree<Integer> kdTree = new KDTree<>(classFeatures,
                classIndices.toArray(new Integer[0]));

        // Generate synthetic samples
        for (int i = 0; i < samplesToGenerate; i++) {
            // Randomly select base sample
            int randomIndex = random.nextInt(classIndices.size());
            int baseInstanceIndex = classIndices.get(randomIndex);
            double[] baseInstance = features[baseInstanceIndex];

            // Find k+1 nearest neighbors (including the sample itself)
            int actualK = Math.min(k + 1, classIndices.size());
            smile.neighbor.Neighbor<double[], Integer>[] neighbors = kdTree.search(baseInstance, actualK);

            // Select random neighbor (excluding the sample itself)
            smile.neighbor.Neighbor<double[], Integer> randomNeighbor;
            do {
                int neighborIndex = random.nextInt(neighbors.length);
                randomNeighbor = neighbors[neighborIndex];
            } while (randomNeighbor.value().equals(baseInstanceIndex) && neighbors.length > 1);

            // Generate synthetic sample
            double[] neighborInstance = features[randomNeighbor.value()];
            double[] syntheticSample = generateSyntheticSample(baseInstance, neighborInstance);

            syntheticSamples.add(syntheticSample);
        }

        return syntheticSamples;
    }

    /**
     * Generate one synthetic sample between two samples using linear interpolation
     * @param sample1 first sample
     * @param sample2 second sample
     * @return synthetic sample
     */
    private double[] generateSyntheticSample(double[] sample1, double[] sample2) {
        double[] synthetic = new double[sample1.length];

        for (int i = 0; i < sample1.length; i++) {
            // Random coefficient between 0 and 1
            double lambda = random.nextDouble();
            // Linear interpolation
            synthetic[i] = sample1[i] + lambda * (sample2[i] - sample1[i]);
        }

        return synthetic;
    }

    /**
     * Analyze class distribution and return mapping of class labels to sample indices
     * @param labels array of class labels
     * @return map from class label to list of sample indices
     */
    private Map<Integer, List<Integer>> analyzeClassDistribution(int[] labels) {
        Map<Integer, List<Integer>> classIndices = new HashMap<>();

        for (int i = 0; i < labels.length; i++) {
            classIndices.computeIfAbsent(labels[i], k -> new ArrayList<>()).add(i);
        }

        return classIndices;
    }

    /**
     * Set number of nearest neighbors
     * @param k number of neighbors (must be > 0)
     */
    public void setK(int k) {
        if (k <= 0) {
            throw new IllegalArgumentException("k must be greater than 0");
        }
        this.k = k;
    }

    /**
     * Get current value of k
     * @return number of nearest neighbors
     */
    public int getK() {
        return k;
    }

    /**
     * Set seed for random number generator to ensure reproducibility
     * @param seed random seed
     */
    public void setSeed(long seed) {
        this.random = new Random(seed);
    }
}
