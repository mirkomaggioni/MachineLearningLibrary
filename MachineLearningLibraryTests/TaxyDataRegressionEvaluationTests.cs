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
		private readonly PredictionService predictionService = new PredictionService();
		private readonly string _dataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\traindata\taxi.csv";
		private readonly string _testDataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\testdata\taxi.csv";
		private readonly char _separator = ',';
		private readonly string _predictedColumn = "FareAmount";
		private readonly string[] _concatenatedColumns = new[] { "VendorId", "RateCode", "PassengerCount", "TripDistance", "PaymentType" };

		[Test]
		public void TaxyDataRegressionEvaluationTest()
		{
			var pipelineParameters = GetPipelineParameters(_dataPath);
			var pipelineTestParameters = GetPipelineParameters(_testDataPath);

			var model = predictionService.Train<TaxyData, TaxyTripFarePrediction>(pipelineParameters, AlgorithmType.StochasticDualCoordinateAscentRegressor);
			var result = predictionService.EvaluateRegression(model, pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.StochasticDualCoordinateAscentRegressor), result);

			model = predictionService.Train<TaxyData, TaxyTripFarePrediction>(pipelineParameters, AlgorithmType.FastTreeRegressor);
			result = predictionService.EvaluateRegression(model, pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.FastTreeRegressor), result);

			model = predictionService.Train<TaxyData, TaxyTripFarePrediction>(pipelineParameters, AlgorithmType.FastTreeTweedieRegressor);
			result = predictionService.EvaluateRegression(model, pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.FastTreeTweedieRegressor), result);

			model = predictionService.Train<TaxyData, TaxyTripFarePrediction>(pipelineParameters, AlgorithmType.FastForestRegressor);
			result = predictionService.EvaluateRegression(model, pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.FastForestRegressor), result);

			model = predictionService.Train<TaxyData, TaxyTripFarePrediction>(pipelineParameters, AlgorithmType.OnlineGradientDescentRegressor);
			result = predictionService.EvaluateRegression(model, pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.OnlineGradientDescentRegressor), result);

			model = predictionService.Train<TaxyData, TaxyTripFarePrediction>(pipelineParameters, AlgorithmType.PoissonRegressor);
			result = predictionService.EvaluateRegression(model, pipelineParameters, pipelineTestParameters);
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
			return new PipelineParameters<TaxyData>(dataPath, _separator, _predictedColumn, null, _concatenatedColumns);
		}
	}
}
