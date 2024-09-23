using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace SentimentAnalysis
{
    // Data class for sentiment analysis
    public class SentimentData
    {
        [LoadColumn(0), ColumnName("Label")]
        public bool Sentiment { get; set; } // True for positive, False for negative

        [LoadColumn(1)]
        public string Text { get; set; }
    }

    // Class for predictions
    public class SentimentPrediction
    {
        [ColumnName("PredictedLabel")]
        public bool Sentiment { get; set; } // True for positive, False for negative

        [ColumnName("Score")]
        public float Probability { get; set; } // Probability of the predicted sentiment
    }

    class Program
    {
        static void Main(string[] args)
        {
            // 1. Load the data
            MLContext mlContext = new MLContext();
            string dataPath = "sentiment_data.csv"; // Replace with your data file path
            IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentData>(dataPath, hasHeader: true, separatorChar: ',');

            // 2. Define the training pipeline
            var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", "Text")
                .Append(mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features"))
                .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"))
                .Append(mlContext.Transforms.Conversion.MapValueToKey("Score", "Score"));

            // 3. Train the model
            ITransformer model = pipeline.Fit(dataView);

            // 4. Create a prediction engine
            var predictionEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

            // 5. Make a prediction
            string newText = "This is a great product! I love it."; // Example text
            SentimentData newSentiment = new SentimentData { Text = newText };

            SentimentPrediction prediction = predictionEngine.Predict(newSentiment);

            // 6. Display the prediction
            Console.WriteLine($"Predicted Sentiment: {(prediction.Sentiment ? "Positive" : "Negative")}");
            Console.WriteLine($"Probability: {prediction.Probability}");

            Console.ReadKey();
        }
    }
}
