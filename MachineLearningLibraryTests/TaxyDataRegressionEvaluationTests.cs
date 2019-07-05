using System;
using System.IO;
using System.Reflection;
using MachineLearningLibrary.Interfaces;
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
		private readonly string[] _concatenatedColumns = new[] { "VendorId", "RateCode", "PassengerCount", "TripDistance", "PaymentType" };

		[Test]
		public void TaxyDataRegressionEvaluationTest()
		{
			var pipelineParameters = GetPipelineParameters(_dataPath);
			var pipelineTestParameters = new Pipeline<TaxyData>(_testDataPath, _separator);

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
			Console.WriteLine($"LossFn = {regressionMetrics.LossFunction}");
			Console.WriteLine($"Rms = {regressionMetrics.RootMeanSquaredError}");
			Console.WriteLine($"RSquared = {regressionMetrics.RSquared}");
			Console.WriteLine($"------------- {algorithm} - END EVALUATION -------------");
		}

		private ITrain GetPipelineParameters(string dataPath)
		{
			return (new Pipeline<CarData>(dataPath, _separator))
				.CopyColumn("Label", "FareAmount")
				.ConcatenateColumns(_concatenatedColumns);
		}
	}
}
