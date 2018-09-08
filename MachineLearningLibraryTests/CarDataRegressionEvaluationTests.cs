using System;
using System.IO;
using System.Reflection;
using System.Threading.Tasks;
using MachineLearningLibrary.Models;
using MachineLearningLibrary.Services;
using Microsoft.ML;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using NUnit.Framework;

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
		public async Task CarDataRegressionEvaluationTest()
		{
			var stochasticDualCoordinateAscentRegressorAlgorithm = new StochasticDualCoordinateAscentRegressor();
			stochasticDualCoordinateAscentRegressorAlgorithm.BiasLearningRate = 0.1f;
			stochasticDualCoordinateAscentRegressorAlgorithm.CheckFrequency = 0;
			stochasticDualCoordinateAscentRegressorAlgorithm.ConvergenceTolerance = 0.1f;

			var pipelineParameters = GetPipelineParameters(stochasticDualCoordinateAscentRegressorAlgorithm);
			var pipelineTestParameters = GetPipelineTestParameters();
			var modelPath = await predictionService.TrainAsync<CarData, CarPricePrediction>(pipelineParameters);
			var result = await predictionService.EvaluateRegressionAsync<CarData, CarPricePrediction>(pipelineTestParameters, modelPath);
			LogResult(nameof(StochasticDualCoordinateAscentRegressor), result);

			pipelineParameters = GetPipelineParameters(new FastTreeRegressor());
			modelPath = await predictionService.TrainAsync<CarData, CarPricePrediction>(pipelineParameters);
			result = await predictionService.EvaluateRegressionAsync<CarData, CarPricePrediction>(pipelineTestParameters, modelPath);
			LogResult(nameof(FastTreeRegressor), result);

			pipelineParameters = GetPipelineParameters(new FastTreeTweedieRegressor());
			modelPath = await predictionService.TrainAsync<CarData, CarPricePrediction>(pipelineParameters);
			result = await predictionService.EvaluateRegressionAsync<CarData, CarPricePrediction>(pipelineTestParameters, modelPath);
			LogResult(nameof(FastTreeTweedieRegressor), result);

			pipelineParameters = GetPipelineParameters(new FastForestRegressor());
			modelPath = await predictionService.TrainAsync<CarData, CarPricePrediction>(pipelineParameters);
			result = await predictionService.EvaluateRegressionAsync<CarData, CarPricePrediction>(pipelineTestParameters, modelPath);
			LogResult(nameof(FastForestRegressor), result);

			pipelineParameters = GetPipelineParameters(new OnlineGradientDescentRegressor());
			modelPath = await predictionService.TrainAsync<CarData, CarPricePrediction>(pipelineParameters);
			result = await predictionService.EvaluateRegressionAsync<CarData, CarPricePrediction>(pipelineTestParameters, modelPath);
			LogResult(nameof(OnlineGradientDescentRegressor), result);

			pipelineParameters = GetPipelineParameters(new PoissonRegressor());
			modelPath = await predictionService.TrainAsync<CarData, CarPricePrediction>(pipelineParameters);
			result = await predictionService.EvaluateRegressionAsync<CarData, CarPricePrediction>(pipelineTestParameters, modelPath);
			LogResult(nameof(PoissonRegressor), result);
		}

		private void LogResult(string algorithm, RegressionMetrics regressionMetrics)
		{
			Console.WriteLine($"------------- {algorithm} - EVALUATION RESULTS -------------");
			Console.WriteLine($"RMS = {regressionMetrics.Rms}");
			Console.WriteLine($"RSQUARED = {regressionMetrics.RSquared}");
			Console.WriteLine($"------------- {algorithm} - END EVALUATION -------------");
		}

		private PipelineParameters<CarData> GetPipelineParameters(ILearningPipelineItem algorithm) {
			return new PipelineParameters<CarData>(_dataPath, _separator, null, _alphanumericColumns, null, _concatenatedColumns, algorithm);
		}

		private PipelineParameters<CarData> GetPipelineTestParameters() {
			return new PipelineParameters<CarData>(_testDataPath, _separator);
		}
	}
}
