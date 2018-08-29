using System;
using System.IO;
using System.Reflection;
using System.Threading.Tasks;
using MachineLearningLibrary.Models;
using MachineLearningLibrary.Services;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using NUnit.Framework;

namespace MachineLearningLibraryTests
{
	[TestFixture]
	public class CarDataRegressionEvaluationTests
	{
		private PredictionService predictionService = new PredictionService();
		private PipelineParameters<CarData> _pipelineParameters;
		private PipelineParameters<CarData> _pipelineTestParameters;

		[SetUp]
		public void Setup()
		{
			var dir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
			var dataPath = $@"{dir}\traindata\car.csv";
			var testDataPath = $@"{dir}\testdata\car.csv";
			var separator = ',';
			var alphanumericColumns = new[] { "Make", "FuelType", "Aspiration", "Doors", "BodyStyle", "DriveWheels", "EngineLocation", "EngineType", "NumOfCylinders", "FuelSystem" };
			var concatenatedColumns = new[] { "Symboling", "NormalizedLosses", "Make", "FuelType", "Aspiration", "Doors", "BodyStyle", "DriveWheels", "EngineLocation", "WheelBase", "Length", "Width", "Height", "CurbWeight", "EngineType", "NumOfCylinders", "EngineSize", "FuelSystem",
												"Bore", "Stroke", "CompressionRatio", "HorsePower", "PeakRpm", "CityMpg", "HighwayMpg"};
			_pipelineParameters = new PipelineParameters<CarData>(dataPath, separator, null, alphanumericColumns, null, concatenatedColumns);
			_pipelineTestParameters = new PipelineParameters<CarData>(testDataPath, separator);
		}

		[Test]
		public async Task CarDataRegressionEvaluationTest()
		{
			var modelPath = await predictionService.TrainAsync<CarData, CarPricePrediction, StochasticDualCoordinateAscentRegressor>(_pipelineParameters);
			var result = await predictionService.EvaluateRegressionAsync<CarData, CarPricePrediction>(_pipelineTestParameters, modelPath);
			LogResult(nameof(StochasticDualCoordinateAscentRegressor), result);

			modelPath = await predictionService.TrainAsync<CarData, CarPricePrediction, FastTreeRegressor>(_pipelineParameters);
			result = await predictionService.EvaluateRegressionAsync<CarData, CarPricePrediction>(_pipelineTestParameters, modelPath);
			LogResult(nameof(FastTreeRegressor), result);

			modelPath = await predictionService.TrainAsync<CarData, CarPricePrediction, FastTreeTweedieRegressor>(_pipelineParameters);
			result = await predictionService.EvaluateRegressionAsync<CarData, CarPricePrediction>(_pipelineTestParameters, modelPath);
			LogResult(nameof(FastTreeTweedieRegressor), result);

			modelPath = await predictionService.TrainAsync<CarData, CarPricePrediction, FastForestRegressor>(_pipelineParameters);
			result = await predictionService.EvaluateRegressionAsync<CarData, CarPricePrediction>(_pipelineTestParameters, modelPath);
			LogResult(nameof(FastForestRegressor), result);

			modelPath = await predictionService.TrainAsync<CarData, CarPricePrediction, OnlineGradientDescentRegressor>(_pipelineParameters);
			result = await predictionService.EvaluateRegressionAsync<CarData, CarPricePrediction>(_pipelineTestParameters, modelPath);
			LogResult(nameof(OnlineGradientDescentRegressor), result);

			modelPath = await predictionService.TrainAsync<CarData, CarPricePrediction, PoissonRegressor>(_pipelineParameters);
			result = await predictionService.EvaluateRegressionAsync<CarData, CarPricePrediction>(_pipelineTestParameters, modelPath);
			LogResult(nameof(PoissonRegressor), result);
		}

		private void LogResult(string algorithm, RegressionMetrics regressionMetrics)
		{
			Console.WriteLine($"------------- {algorithm} - EVALUATION RESULTS -------------");
			Console.WriteLine($"RMS = {regressionMetrics.Rms}");
			Console.WriteLine($"RSQUARED = {regressionMetrics.RSquared}");
			Console.WriteLine($"------------- {algorithm} - END EVALUATION -------------");
		}
	}
}
