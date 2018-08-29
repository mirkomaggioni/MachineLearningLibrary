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
	public class TaxyDataRegressionEvaluationTests
	{
		private PredictionService predictionService = new PredictionService();
		private PipelineParameters<TaxyData> _pipelineParameters;
		private PipelineParameters<TaxyData> _pipelineTestParameters;

		[SetUp]
		public void Setup()
		{
			var dir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
			var dataPath = $@"{dir}\traindata\taxi.csv";
			var testDataPath = $@"{dir}\testdata\taxi.csv";
			var separator = ',';
			var alphanumericColumns = new[] { "VendorId", "RateCode", "PaymentType" };
			var concatenatedColumns = new[] { "VendorId", "RateCode", "PassengerCount", "TripDistance", "PaymentType" };
			_pipelineParameters = new PipelineParameters<TaxyData>(dataPath, separator, null, alphanumericColumns, null, concatenatedColumns);
			_pipelineTestParameters = new PipelineParameters<TaxyData>(testDataPath, separator);
		}

		[Test]
		public async Task TaxyDataRegressionEvaluationTest()
		{
			var modelPath = await predictionService.TrainAsync<TaxyData, TaxyTripFarePrediction, StochasticDualCoordinateAscentRegressor>(_pipelineParameters);
			var result = await predictionService.EvaluateRegressionAsync<TaxyData, TaxyTripFarePrediction>(_pipelineTestParameters, modelPath);
			LogResult(nameof(StochasticDualCoordinateAscentRegressor), result);

			modelPath = await predictionService.TrainAsync<TaxyData, TaxyTripFarePrediction, FastTreeRegressor>(_pipelineParameters);
			result = await predictionService.EvaluateRegressionAsync<TaxyData, TaxyTripFarePrediction>(_pipelineTestParameters, modelPath);
			LogResult(nameof(FastTreeRegressor), result);

			modelPath = await predictionService.TrainAsync<TaxyData, TaxyTripFarePrediction, FastTreeTweedieRegressor>(_pipelineParameters);
			result = await predictionService.EvaluateRegressionAsync<TaxyData, TaxyTripFarePrediction>(_pipelineTestParameters, modelPath);
			LogResult(nameof(FastTreeTweedieRegressor), result);

			modelPath = await predictionService.TrainAsync<TaxyData, TaxyTripFarePrediction, FastForestRegressor>(_pipelineParameters);
			result = await predictionService.EvaluateRegressionAsync<TaxyData, TaxyTripFarePrediction>(_pipelineTestParameters, modelPath);
			LogResult(nameof(FastForestRegressor), result);

			modelPath = await predictionService.TrainAsync<TaxyData, TaxyTripFarePrediction, OnlineGradientDescentRegressor>(_pipelineParameters);
			result = await predictionService.EvaluateRegressionAsync<TaxyData, TaxyTripFarePrediction>(_pipelineTestParameters, modelPath);
			LogResult(nameof(OnlineGradientDescentRegressor), result);

			modelPath = await predictionService.TrainAsync<TaxyData, TaxyTripFarePrediction, PoissonRegressor>(_pipelineParameters);
			result = await predictionService.EvaluateRegressionAsync<TaxyData, TaxyTripFarePrediction>(_pipelineTestParameters, modelPath);
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
