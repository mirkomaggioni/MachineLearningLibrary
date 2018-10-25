using System;
using System.IO;
using System.Reflection;
using System.Threading.Tasks;
using MachineLearningLibrary.Models;
using MachineLearningLibrary.Services;
using Microsoft.ML.Legacy;
using Microsoft.ML.Legacy.Models;
using Microsoft.ML.Legacy.Trainers;
using NUnit.Framework;

namespace MachineLearningLibraryTests
{
	[TestFixture]
	public class TaxyDataRegressionEvaluationTests
	{
		private PredictionService predictionService = new PredictionService();
		private string _dataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\traindata\taxi.csv";
		private string _testDataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\testdata\taxi.csv";
		private char _separator = ',';
		private string[] _alphanumericColumns = new[] { "VendorId", "RateCode", "PaymentType" };
		private string[] _concatenatedColumns = new[] { "VendorId", "RateCode", "PassengerCount", "TripDistance", "PaymentType" };

		[Test]
		public async Task TaxyDataRegressionEvaluationTest()
		{
			var pipelineParameters = GetPipelineParameters(new StochasticDualCoordinateAscentRegressor());
			var pipelineTestParameters = GetPipelineTestParameters();
			var modelPath = await predictionService.TrainAsync<TaxyData, TaxyTripFarePrediction>(pipelineParameters);
			var result = await predictionService.EvaluateRegressionAsync<TaxyData, TaxyTripFarePrediction>(pipelineTestParameters, modelPath);
			LogResult(nameof(StochasticDualCoordinateAscentRegressor), result);

			pipelineParameters = GetPipelineParameters(new FastTreeRegressor());
			modelPath = await predictionService.TrainAsync<TaxyData, TaxyTripFarePrediction>(pipelineParameters);
			result = await predictionService.EvaluateRegressionAsync<TaxyData, TaxyTripFarePrediction>(pipelineTestParameters, modelPath);
			LogResult(nameof(FastTreeRegressor), result);

			pipelineParameters = GetPipelineParameters(new FastTreeTweedieRegressor());
			modelPath = await predictionService.TrainAsync<TaxyData, TaxyTripFarePrediction>(pipelineParameters);
			result = await predictionService.EvaluateRegressionAsync<TaxyData, TaxyTripFarePrediction>(pipelineTestParameters, modelPath);
			LogResult(nameof(FastTreeTweedieRegressor), result);

			pipelineParameters = GetPipelineParameters(new FastForestRegressor());
			modelPath = await predictionService.TrainAsync<TaxyData, TaxyTripFarePrediction>(pipelineParameters);
			result = await predictionService.EvaluateRegressionAsync<TaxyData, TaxyTripFarePrediction>(pipelineTestParameters, modelPath);
			LogResult(nameof(FastForestRegressor), result);

			pipelineParameters = GetPipelineParameters(new OnlineGradientDescentRegressor());
			modelPath = await predictionService.TrainAsync<TaxyData, TaxyTripFarePrediction>(pipelineParameters);
			result = await predictionService.EvaluateRegressionAsync<TaxyData, TaxyTripFarePrediction>(pipelineTestParameters, modelPath);
			LogResult(nameof(OnlineGradientDescentRegressor), result);

			pipelineParameters = GetPipelineParameters(new PoissonRegressor());
			modelPath = await predictionService.TrainAsync<TaxyData, TaxyTripFarePrediction>(pipelineParameters);
			result = await predictionService.EvaluateRegressionAsync<TaxyData, TaxyTripFarePrediction>(pipelineTestParameters, modelPath);
			LogResult(nameof(PoissonRegressor), result);
		}

		private void LogResult(string algorithm, RegressionMetrics regressionMetrics)
		{
			Console.WriteLine($"------------- {algorithm} - EVALUATION RESULTS -------------");
			Console.WriteLine($"RMS = {regressionMetrics.Rms}");
			Console.WriteLine($"RSQUARED = {regressionMetrics.RSquared}");
			Console.WriteLine($"------------- {algorithm} - END EVALUATION -------------");
		}

		private PipelineParameters<TaxyData> GetPipelineParameters(ILearningPipelineItem algorithm) {
			return new PipelineParameters<TaxyData>(_dataPath, _separator, null, _alphanumericColumns, null, _concatenatedColumns, algorithm);
		}

		private PipelineParameters<TaxyData> GetPipelineTestParameters() {
			return new PipelineParameters<TaxyData>(_testDataPath, _separator);
		}
	}
}
