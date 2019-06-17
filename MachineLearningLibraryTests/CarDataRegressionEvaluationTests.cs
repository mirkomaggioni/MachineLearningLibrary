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
			var pipelineTestParameters = GetPipelineParameters(_testDataPath);

			var model = predictionService.Train<CarData, CarPricePrediction>(pipelineParameters, AlgorithmType.StochasticDualCoordinateAscentRegressor);
			var result = predictionService.EvaluateRegression(model, pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.StochasticDualCoordinateAscentRegressor), result);

			model = predictionService.Train<CarData, CarPricePrediction>(pipelineParameters, AlgorithmType.FastTreeRegressor);
			result = predictionService.EvaluateRegression(model, pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.FastTreeRegressor), result);

			//model = predictionService.Train<CarData, CarPricePrediction>(pipelineParameters, AlgorithmType.FastTreeTweedieRegressor);
			//result = predictionService.EvaluateRegression(model, pipelineParameters, pipelineTestParameters);
			//LogResult(nameof(AlgorithmType.FastTreeTweedieRegressor), result);

			//model = predictionService.Train<CarData, CarPricePrediction>(pipelineParameters, AlgorithmType.FastForestRegressor);
			//result = predictionService.EvaluateRegression(model, pipelineParameters, pipelineTestParameters);
			//LogResult(nameof(AlgorithmType.FastForestRegressor), result);

			//model = predictionService.Train<CarData, CarPricePrediction>(pipelineParameters, AlgorithmType.OnlineGradientDescentRegressor);
			//result = predictionService.EvaluateRegression(model, pipelineParameters, pipelineTestParameters);
			//LogResult(nameof(AlgorithmType.OnlineGradientDescentRegressor), result);

			//model = predictionService.Train<CarData, CarPricePrediction>(pipelineParameters, AlgorithmType.PoissonRegressor);
			//result = predictionService.EvaluateRegression(model, pipelineParameters, pipelineTestParameters);
			//LogResult(nameof(AlgorithmType.PoissonRegressor), result);
		}

		private void LogResult(string algorithm, RegressionMetrics regressionMetrics)
		{
			Console.WriteLine($"------------- {algorithm} - EVALUATION RESULTS -------------");
			Console.WriteLine($"RMS = {regressionMetrics.RootMeanSquaredError}");
			Console.WriteLine($"RSQUARED = {regressionMetrics.RSquared}");
			Console.WriteLine($"------------- {algorithm} - END EVALUATION -------------");
		}

		private PipelineParameters<CarData> GetPipelineParameters(string dataPath)
		{
			return new PipelineParameters<CarData>(dataPath, _separator, (_predictedColumn, true), _concatenatedColumns, _alphanumericColumns);
		}
	}
}
