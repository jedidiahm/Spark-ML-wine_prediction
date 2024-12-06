package com.example;

import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.types.StructType;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.StandardScaler;
import org.apache.spark.ml.classification.GBTClassificationModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.classification.RandomForestClassificationModel;

import java.util.Arrays;

public class Test {
    public static void main(String[] args) {
        // Configure input paths with default S3 locations if not provided
        String modelPath = args.length > 0 ? args[0] : "s3://spark-wine-pred/top_model/";
        String testFilePath = args.length > 1 ? args[1] : "s3://spark-wine-pred/ValidationDataset.csv";

        // Create Spark session for distributed processing
        SparkSession spark = SparkSession.builder().appName("cloud assignment test").getOrCreate();

        // Prepare validation dataset with feature scaling
        Dataset<Row> validationData = loadAndProcessData(spark, testFilePath);
        validationData.select("scaledFeatures").show(false);

        // Load pre-trained Random Forest model and generate predictions
        RandomForestClassificationModel loadedModel = RandomForestClassificationModel.load(modelPath);
        Dataset<Row> predictions = loadedModel.transform(validationData);

        // Display prediction results
        predictions.show();

        // Calculate model performance using F1 score
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("quality")
                .setPredictionCol("prediction")
                .setMetricName("f1");
        double score = evaluator.evaluate(predictions);
        System.out.println("F1 Score = " + score);

        // Clean up Spark resources
        spark.stop();
    }

    /**
     * Loads and preprocesses the wine quality dataset.
     * @param spark Active SparkSession
     * @param filePath Path to the CSV file
     * @return Processed Dataset with scaled features
     */
    private static Dataset<Row> loadAndProcessData(SparkSession spark, String filePath) {
        // Load CSV file with wine quality data
        Dataset<Row> df = spark.read().option("header", "true").option("sep", ";").option("inferSchema", "true")
                .csv(filePath);

        // Define column names for the wine quality features
        String[] columns = new String[] {
                "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
                "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
                "pH", "sulphates", "alcohol", "quality"
        };
        df = df.toDF(columns);

        // Create feature vector from input columns
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(Arrays.copyOfRange(columns, 0, columns.length - 1))
                .setOutputCol("features");

        // Scale features to have zero mean and unit variance
        StandardScaler scaler = new StandardScaler()
                .setInputCol("features")
                .setOutputCol("scaledFeatures")
                .setWithStd(true)
                .setWithMean(true);

        // Create and execute pipeline for feature processing
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {assembler, scaler});
        PipelineModel pipelineModel = pipeline.fit(df);
        return pipelineModel.transform(df);
    }
}
