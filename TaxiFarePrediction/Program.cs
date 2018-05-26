using Microsoft.ML;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System;
using System.Threading.Tasks;

namespace TaxiFarePrediction
{
    class Program
    {
        const string _dataPath = @".\Data\taxi-fare-test.csv";
        const string _testDataPath = @".\Data\taxi-fare-train.csv";
        const string _modelPath = @".\Models\Model.zip";

        static async Task Main(string[] args)
        {
            PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = await Train();
            Evaluate(model);

            var prediction = model.Predict(TestTrip.Trip1);
            Console.WriteLine("Predicted fare: {0}, actual fare; 29.5", prediction.fare_amount); ;
        }

        private static void Evaluate(PredictionModel<TaxiTrip, TaxiTripFarePrediction> model)
        {
            var testData = new TextLoader<TaxiTrip>(_testDataPath, useHeader: true, separator: ",");
            var evaluator = new RegressionEvaluator();
            RegressionMetrics metrics = evaluator.Evaluate(model, testData);
            Console.WriteLine("Rms=" + metrics.Rms);
            Console.WriteLine("RSquared = " + metrics.RSquared);
        }

        private static async Task<PredictionModel<TaxiTrip, TaxiTripFarePrediction>> Train()
        {
            var pipeline = new LearningPipeline
            {
                new TextLoader<TaxiTrip>(_dataPath, useHeader: true, separator: ","),
                new ColumnCopier(("fare_amount", "Label")),
                new CategoricalHashOneHotVectorizer("vendor_id", "rate_code", "payment_type"),
                new ColumnConcatenator("Features", "vendor_id", "rate_code", "passenger_count", "trip_distance", "payment_type"),
                new FastTreeRegressor()
            };

            PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = pipeline.Train<TaxiTrip, TaxiTripFarePrediction>();
            await model.WriteAsync(_modelPath);
            return model;
        }
    }
}
