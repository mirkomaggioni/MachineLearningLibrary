using System;
using System.IO;
using System.Reflection;
using MachineLearningLibrary.Models;
using MachineLearningLibrary.Services;
using Microsoft.ML.Data;
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
		private string[] _concatenatedColumns = new[] { "VendorId", "RateCode", "PassengerCount", "TripDistance", "PaymentType" };

		[Test]
		public void TaxyDataRegressionEvaluationTest()
		{
			var pipelineParameters = GetPipelineParameters(_dataPath);
			var pipelineTestParameters = GetPipelineParameters(_testDataPath);

			var modelPath = predictionService.Train<TaxyData, TaxyTripFarePrediction>(pipelineParameters, AlgorithmType.StochasticDualCoordinateAscentRegressor);
			var result = predictionService.EvaluateRegression(pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.StochasticDualCoordinateAscentRegressor), result);

			modelPath = predictionService.Train<TaxyData, TaxyTripFarePrediction>(pipelineParameters, AlgorithmType.FastTreeRegressor);
			result = predictionService.EvaluateRegression(pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.FastTreeRegressor), result);

			modelPath = predictionService.Train<TaxyData, TaxyTripFarePrediction>(pipelineParameters, AlgorithmType.FastTreeTweedieRegressor);
			result = predictionService.EvaluateRegression(pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.FastTreeTweedieRegressor), result);

			modelPath = predictionService.Train<TaxyData, TaxyTripFarePrediction>(pipelineParameters, AlgorithmType.FastForestRegressor);
			result = predictionService.EvaluateRegression(pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.FastForestRegressor), result);

			modelPath = predictionService.Train<TaxyData, TaxyTripFarePrediction>(pipelineParameters, AlgorithmType.OnlineGradientDescentRegressor);
			result = predictionService.EvaluateRegression(pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.OnlineGradientDescentRegressor), result);

			modelPath = predictionService.Train<TaxyData, TaxyTripFarePrediction>(pipelineParameters, AlgorithmType.PoissonRegressor);
			result = predictionService.EvaluateRegression(pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.PoissonRegressor), result);
		}

		private void LogResult(string algorithm, RegressionMetrics regressionMetrics)
		{
			Console.WriteLine($"------------- {algorithm} - EVALUATION RESULTS -------------");
			Console.WriteLine($"L1 = {regressionMetrics.L1}");
			Console.WriteLine($"L2 = {regressionMetrics.L2}");
			Console.WriteLine($"LossFn = {regressionMetrics.LossFn}");
			Console.WriteLine($"Rms = {regressionMetrics.Rms}");
			Console.WriteLine($"RSquared = {regressionMetrics.RSquared}");
			Console.WriteLine($"------------- {algorithm} - END EVALUATION -------------");
		}

		private PipelineParameters<TaxyData> GetPipelineParameters(string dataPath)
		{
			return new PipelineParameters<TaxyData>(dataPath, _separator, null, _concatenatedColumns);
		}
	}
}
