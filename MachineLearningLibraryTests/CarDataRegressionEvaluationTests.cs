using System;
using System.IO;
using System.Reflection;
using MachineLearningLibrary.Models;
using MachineLearningLibrary.Services;
using NUnit.Framework;
using Microsoft.ML.Data;
using MachineLearningLibrary.Interfaces;

namespace MachineLearningLibraryTests
{
	[TestFixture]
	public class CarDataRegressionEvaluationTests
	{
		private readonly PredictionService predictionService = new PredictionService();
		private readonly string _dataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\traindata\car.csv";
		private readonly string _testDataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\testdata\car.csv";
		private readonly char _separator = ',';
		private readonly string _predictedColumn = "Price";
		private readonly string[] _alphanumericColumns = new[] { "Make", "FuelType", "Aspiration", "Doors", "BodyStyle", "DriveWheels", "EngineLocation", "EngineType", "NumOfCylinders", "FuelSystem" };
		private readonly string[] _concatenatedColumns = new[] { "Symboling", "NormalizedLosses", "Make", "FuelType", "Aspiration", "Doors", "BodyStyle", "DriveWheels", "EngineLocation", "WheelBase", "Length", "Width", "Height", "CurbWeight", "EngineType", "NumOfCylinders", "EngineSize", "FuelSystem",
												"Bore", "Stroke", "CompressionRatio", "HorsePower", "PeakRpm", "CityMpg", "HighwayMpg"};

		[Test]
		public void CarDataRegressionEvaluationTest()
		{
			var pipeline = new Pipeline2<CarData>(_dataPath, _separator, AlgorithmType.StochasticDualCoordinateAscentRegressor, (_predictedColumn, false, null), _concatenatedColumns, _alphanumericColumns);
			var pipelineTest = new Pipeline2<CarData>(_testDataPath, _separator);
			pipeline.BuildModel();

			var result = pipeline.EvaluateRegression(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.StochasticDualCoordinateAscentRegressor), result);

			pipeline = new Pipeline2<CarData>(_dataPath, _separator, AlgorithmType.FastTreeRegressor, (_predictedColumn, false, null), _concatenatedColumns, _alphanumericColumns);
			pipeline.BuildModel();
			result = pipeline.EvaluateRegression(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.FastTreeRegressor), result);

			pipeline = new Pipeline2<CarData>(_dataPath, _separator, AlgorithmType.FastTreeTweedieRegressor, (_predictedColumn, false, null), _concatenatedColumns, _alphanumericColumns);
			pipeline.BuildModel();
			result = pipeline.EvaluateRegression(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.FastTreeTweedieRegressor), result);

			pipeline = new Pipeline2<CarData>(_dataPath, _separator, AlgorithmType.FastForestRegressor, (_predictedColumn, false, null), _concatenatedColumns, _alphanumericColumns);
			pipeline.BuildModel();
			result = pipeline.EvaluateRegression(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.FastForestRegressor), result);

			//pipeline = new Pipeline2<CarData>(_dataPath, _separator, AlgorithmType.OnlineGradientDescentRegressor, (_predictedColumn, false), _concatenatedColumns, _alphanumericColumns);
			//pipeline.BuildModel();
			//result = pipeline.EvaluateRegression(pipelineTest.DataView);
			//LogResult(nameof(AlgorithmType.OnlineGradientDescentRegressor), result);

			pipeline = new Pipeline2<CarData>(_dataPath, _separator, AlgorithmType.PoissonRegressor, (_predictedColumn, false, null), _concatenatedColumns, _alphanumericColumns);
			pipeline.BuildModel();
			result = pipeline.EvaluateRegression(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.PoissonRegressor), result);
		}

		private void LogResult(string algorithm, RegressionMetrics regressionMetrics)
		{
			Console.WriteLine($"------------- {algorithm} - EVALUATION RESULTS -------------");
			Console.WriteLine($"RMS = {regressionMetrics.RootMeanSquaredError}");
			Console.WriteLine($"RSQUARED = {regressionMetrics.RSquared}");
			Console.WriteLine($"------------- {algorithm} - END EVALUATION -------------");
		}
	}
}
