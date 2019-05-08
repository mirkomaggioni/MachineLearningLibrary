using System;
using System.IO;
using System.Reflection;
using MachineLearningLibrary.Models;
using MachineLearningLibrary.Services;
using NUnit.Framework;
using Microsoft.ML.Data;

namespace MachineLearningLibraryTests
{
	[TestFixture]
	public class CarDataRegressionEvaluationTests
	{
		private PredictionService predictionService = new PredictionService();
		private string _dataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\traindata\car.csv";
		private string _testDataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\testdata\car.csv";
		private char _separator = ',';
		private string[] _alphanumericColumns = new[] { "Make", "FuelType", "Aspiration", "Doors", "BodyStyle", "DriveWheels", "EngineLocation", "EngineType", "NumOfCylinders", "FuelSystem" };
		private string[] _concatenatedColumns = new[] { "Symboling", "NormalizedLosses", "Make", "FuelType", "Aspiration", "Doors", "BodyStyle", "DriveWheels", "EngineLocation", "WheelBase", "Length", "Width", "Height", "CurbWeight", "EngineType", "NumOfCylinders", "EngineSize", "FuelSystem",
												"Bore", "Stroke", "CompressionRatio", "HorsePower", "PeakRpm", "CityMpg", "HighwayMpg"};

		[Test]
		public void CarDataRegressionEvaluationTest()
		{
			var pipelineParameters = GetPipelineParameters(_dataPath);
			var pipelineTestParameters = GetPipelineParameters(_testDataPath);

			var modelPath = predictionService.Train<CarData, CarPricePrediction>(pipelineParameters, AlgorithmType.StochasticDualCoordinateAscent);
			var result = predictionService.EvaluateRegression(pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.StochasticDualCoordinateAscent), result);

			modelPath = predictionService.Train<CarData, CarPricePrediction>(pipelineParameters, AlgorithmType.FastTree);
			result = predictionService.EvaluateRegression(pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.FastTree), result);

			modelPath = predictionService.Train<CarData, CarPricePrediction>(pipelineParameters, AlgorithmType.FastTreeTweedieRegressor);
			result = predictionService.EvaluateRegression(pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.FastTreeTweedieRegressor), result);

			modelPath = predictionService.Train<CarData, CarPricePrediction>(pipelineParameters, AlgorithmType.FastForestRegressor);
			result = predictionService.EvaluateRegression(pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.FastForestRegressor), result);

			modelPath = predictionService.Train<CarData, CarPricePrediction>(pipelineParameters, AlgorithmType.OnlineGradientDescentRegressor);
			result = predictionService.EvaluateRegression(pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.OnlineGradientDescentRegressor), result);

			modelPath = predictionService.Train<CarData, CarPricePrediction>(pipelineParameters, AlgorithmType.PoissonRegressor);
			result = predictionService.EvaluateRegression(pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.PoissonRegressor), result);
		}

		private void LogResult(string algorithm, RegressionMetrics regressionMetrics)
		{
			Console.WriteLine($"------------- {algorithm} - EVALUATION RESULTS -------------");
			Console.WriteLine($"RMS = {regressionMetrics.Rms}");
			Console.WriteLine($"RSQUARED = {regressionMetrics.RSquared}");
			Console.WriteLine($"------------- {algorithm} - END EVALUATION -------------");
		}

		private PipelineParameters<CarData> GetPipelineParameters(string dataPath)
		{
			return new PipelineParameters<CarData>(dataPath, _separator, _alphanumericColumns, _concatenatedColumns);
		}
	}
}
