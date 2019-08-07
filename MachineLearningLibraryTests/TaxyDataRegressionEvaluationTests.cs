using System;
using System.IO;
using System.Reflection;
using MachineLearningLibrary.Models;
using MachineLearningLibrary.Models.Data;
using Microsoft.ML.Data;
using NUnit.Framework;

namespace MachineLearningLibraryTests
{
	[TestFixture]
	public class TaxyDataRegressionEvaluationTests
	{
		private readonly string _dataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\traindata\taxi.csv";
		private readonly string _testDataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\testdata\taxi.csv";
		private readonly char _separator = ',';
		private readonly string[] _concatenatedColumns = new[] { "VendorId", "RateCode", "PassengerCount", "TripDistance", "PaymentType" };
		private readonly string[] _alphanumericColumns = new[] { "VendorId", "RateCode", "PaymentType" };
		private readonly string _predictedColumn = "FareAmount";

		[Test]
		public void TaxyDataRegressionEvaluationTest()
		{
			var pipeline = new Pipeline<Taxy>(_dataPath, _separator, AlgorithmType.StochasticDualCoordinateAscentRegressor, new PredictedColumn(_predictedColumn), _concatenatedColumns, _alphanumericColumns);
			var pipelineTest = new Pipeline<Taxy>(_testDataPath, _separator);
			pipeline.BuildModel();

			var result = pipeline.EvaluateRegression(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.StochasticDualCoordinateAscentRegressor), result);

			pipeline = new Pipeline<Taxy>(_dataPath, _separator, AlgorithmType.FastTreeRegressor, new PredictedColumn(_predictedColumn), _concatenatedColumns, _alphanumericColumns);
			pipeline.BuildModel();
			result = pipeline.EvaluateRegression(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.FastTreeRegressor), result);

			pipeline = new Pipeline<Taxy>(_dataPath, _separator, AlgorithmType.FastTreeTweedieRegressor, new PredictedColumn(_predictedColumn), _concatenatedColumns, _alphanumericColumns);
			pipeline.BuildModel();
			result = pipeline.EvaluateRegression(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.FastTreeTweedieRegressor), result);

			pipeline = new Pipeline<Taxy>(_dataPath, _separator, AlgorithmType.FastForestRegressor, new PredictedColumn(_predictedColumn), _concatenatedColumns, _alphanumericColumns);
			pipeline.BuildModel();
			result = pipeline.EvaluateRegression(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.FastForestRegressor), result);

			pipeline = new Pipeline<Taxy>(_dataPath, _separator, AlgorithmType.OnlineGradientDescentRegressor, new PredictedColumn(_predictedColumn), _concatenatedColumns, _alphanumericColumns);
			pipeline.BuildModel();
			result = pipeline.EvaluateRegression(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.OnlineGradientDescentRegressor), result);

			pipeline = new Pipeline<Taxy>(_dataPath, _separator, AlgorithmType.PoissonRegressor, new PredictedColumn(_predictedColumn), _concatenatedColumns, _alphanumericColumns);
			pipeline.BuildModel();
			result = pipeline.EvaluateRegression(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.PoissonRegressor), result);
		}

		[Test]
		[TestCase(1, "CSH", "1", 1.78f, 1200, "VTS")]
		[TestCase(1, "CSH", "1", 1.78f, 1200, "VTS")]
		[TestCase(1, "CSH", "1", 1.9f, 1080, "VTS")]
		[TestCase(1, "CRD", "1", 2.69f, 660, "VTS")]
		public void TaxyDataRegressionPredictTest(int passengerCount, string paymentType, string rateCode, float tripDistance, float tripTime, string vendorId)
		{
			var pipeline = new Pipeline<Taxy>(_dataPath, _separator, AlgorithmType.StochasticDualCoordinateAscentRegressor, new PredictedColumn(_predictedColumn), _concatenatedColumns, _alphanumericColumns);
			pipeline.BuildModel();

			var prediction = pipeline.PredictScore<Taxy, TaxyTripFarePrediction, float>(new Taxy() { PassengerCount = passengerCount, PaymentType = paymentType, RateCode = rateCode, TripDistance = tripDistance, TripTime = tripTime, VendorId = vendorId });

			Assert.IsTrue(prediction.Score > 0);
		}

		private void LogResult(string algorithm, RegressionMetrics regressionMetrics)
		{
			Console.WriteLine($"------------- {algorithm} - EVALUATION RESULTS -------------");
			Console.WriteLine($"LossFn = {regressionMetrics.LossFunction}");
			Console.WriteLine($"Rms = {regressionMetrics.RootMeanSquaredError}");
			Console.WriteLine($"RSquared = {regressionMetrics.RSquared}");
			Console.WriteLine($"------------- {algorithm} - END EVALUATION -------------");
		}
	}
}
