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
			var pipelineParameters = GetPipelineParameters(_dataPath);
			var pipelineTestParameters = new Pipeline<CarData>(_testDataPath, _separator);
			var pipelineTransformer = pipelineParameters.Train(AlgorithmType.StochasticDualCoordinateAscentRegressor);
			var result = pipelineTransformer.EvaluateRegression(pipelineTestParameters.DataView);
			LogResult(nameof(AlgorithmType.StochasticDualCoordinateAscentRegressor), result);

			pipelineTransformer = pipelineParameters.Train(AlgorithmType.FastTreeRegressor);
			result = pipelineTransformer.EvaluateRegression(pipelineTestParameters.DataView);
			LogResult(nameof(AlgorithmType.FastTreeRegressor), result);

			pipelineTransformer = pipelineParameters.Train(AlgorithmType.FastTreeTweedieRegressor);
			result = pipelineTransformer.EvaluateRegression(pipelineTestParameters.DataView);
			LogResult(nameof(AlgorithmType.FastTreeTweedieRegressor), result);

			pipelineTransformer = pipelineParameters.Train(AlgorithmType.FastForestRegressor);
			result = pipelineTransformer.EvaluateRegression(pipelineTestParameters.DataView);
			LogResult(nameof(AlgorithmType.FastForestRegressor), result);

			pipelineTransformer = pipelineParameters.Train(AlgorithmType.OnlineGradientDescentRegressor);
			result = pipelineTransformer.EvaluateRegression(pipelineTestParameters.DataView);
			LogResult(nameof(AlgorithmType.OnlineGradientDescentRegressor), result);

			pipelineTransformer = pipelineParameters.Train(AlgorithmType.PoissonRegressor);
			result = pipelineTransformer.EvaluateRegression(pipelineTestParameters.DataView);
			LogResult(nameof(AlgorithmType.PoissonRegressor), result);
		}

		private void LogResult(string algorithm, RegressionMetrics regressionMetrics)
		{
			Console.WriteLine($"------------- {algorithm} - EVALUATION RESULTS -------------");
			Console.WriteLine($"RMS = {regressionMetrics.RootMeanSquaredError}");
			Console.WriteLine($"RSQUARED = {regressionMetrics.RSquared}");
			Console.WriteLine($"------------- {algorithm} - END EVALUATION -------------");
		}

		private ITrain GetPipelineParameters(string dataPath)
		{
			return (new Pipeline<CarData>(dataPath, _separator))
				.CopyColumn("Label", "Price")
				.ConvertAlphanumericColumns(_alphanumericColumns)
				.ConcatenateColumns(_concatenatedColumns);
		}
	}
}
