using System;
using System.IO;
using System.Reflection;
using MachineLearningLibrary.Models;
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
			var pipeline = new Pipeline<TaxyData>(_dataPath, _separator, AlgorithmType.StochasticDualCoordinateAscentRegressor, (_predictedColumn, false, null), _concatenatedColumns, _alphanumericColumns);
			var pipelineTest = new Pipeline<TaxyData>(_testDataPath, _separator);
			pipeline.BuildModel();

			var result = pipeline.EvaluateRegression(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.StochasticDualCoordinateAscentRegressor), result);

			pipeline = new Pipeline<TaxyData>(_dataPath, _separator, AlgorithmType.FastTreeRegressor, (_predictedColumn, false, null), _concatenatedColumns, _alphanumericColumns);
			pipeline.BuildModel();
			result = pipeline.EvaluateRegression(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.FastTreeRegressor), result);

			pipeline = new Pipeline<TaxyData>(_dataPath, _separator, AlgorithmType.FastTreeTweedieRegressor, (_predictedColumn, false, null), _concatenatedColumns, _alphanumericColumns);
			pipeline.BuildModel();
			result = pipeline.EvaluateRegression(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.FastTreeTweedieRegressor), result);

			pipeline = new Pipeline<TaxyData>(_dataPath, _separator, AlgorithmType.FastForestRegressor, (_predictedColumn, false, null), _concatenatedColumns, _alphanumericColumns);
			pipeline.BuildModel();
			result = pipeline.EvaluateRegression(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.FastForestRegressor), result);

			pipeline = new Pipeline<TaxyData>(_dataPath, _separator, AlgorithmType.OnlineGradientDescentRegressor, (_predictedColumn, false, null), _concatenatedColumns, _alphanumericColumns);
			pipeline.BuildModel();
			result = pipeline.EvaluateRegression(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.OnlineGradientDescentRegressor), result);

			pipeline = new Pipeline<TaxyData>(_dataPath, _separator, AlgorithmType.PoissonRegressor, (_predictedColumn, false, null), _concatenatedColumns, _alphanumericColumns);
			pipeline.BuildModel();
			result = pipeline.EvaluateRegression(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.PoissonRegressor), result);
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
