using System;
using System.IO;
using System.Reflection;
using System.Threading.Tasks;
using MachineLearningLibrary.Models;
using MachineLearningLibrary.Services;
using Microsoft.ML.Trainers;
using NUnit.Framework;

namespace MachineLearningLibraryTests
{
	[TestFixture]
	public class RegressionEvaluationTests
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
		public async Task StochasticDualCoordinateAscentRegressorTestAsync()
		{
			var modelPath = await predictionService.TrainAsync<TaxyData, TaxyTripFarePrediction, StochasticDualCoordinateAscentRegressor>(_pipelineParameters);
			var result = await predictionService.EvaluateAsync<TaxyData, TaxyTripFarePrediction>(_pipelineTestParameters, modelPath);

			Console.WriteLine($"Rms = {result.Rms}");
			Console.WriteLine($"RSquared = {result.RSquared}");
		}

		[Test]
		public async Task FastTreeRegressorTestAsync()
		{
			var modelPath = await predictionService.TrainAsync<TaxyData, TaxyTripFarePrediction, FastTreeRegressor>(_pipelineParameters);
			var result = await predictionService.EvaluateAsync<TaxyData, TaxyTripFarePrediction>(_pipelineTestParameters, modelPath);

			Console.WriteLine($"Rms = {result.Rms}");
			Console.WriteLine($"RSquared = {result.RSquared}");
		}

		[Test]
		public async Task FastTreeTweedieRegressorTestAsync()
		{
			var modelPath = await predictionService.TrainAsync<TaxyData, TaxyTripFarePrediction, FastTreeTweedieRegressor>(_pipelineParameters);
			var result = await predictionService.EvaluateAsync<TaxyData, TaxyTripFarePrediction>(_pipelineTestParameters, modelPath);

			Console.WriteLine($"Rms = {result.Rms}");
			Console.WriteLine($"RSquared = {result.RSquared}");
		}

		[Test]
		public async Task FastForestRegressorTestAsync()
		{
			var modelPath = await predictionService.TrainAsync<TaxyData, TaxyTripFarePrediction, FastForestRegressor>(_pipelineParameters);
			var result = await predictionService.EvaluateAsync<TaxyData, TaxyTripFarePrediction>(_pipelineTestParameters, modelPath);

			Console.WriteLine($"Rms = {result.Rms}");
			Console.WriteLine($"RSquared = {result.RSquared}");
		}

		[Test]
		public async Task OnlineGradientDescentRegressorTestAsync()
		{
			var modelPath = await predictionService.TrainAsync<TaxyData, TaxyTripFarePrediction, OnlineGradientDescentRegressor>(_pipelineParameters);
			var result = await predictionService.EvaluateAsync<TaxyData, TaxyTripFarePrediction>(_pipelineTestParameters, modelPath);

			Console.WriteLine($"Rms = {result.Rms}");
			Console.WriteLine($"RSquared = {result.RSquared}");
		}

		[Test]
		public async Task PoissonRegressorTestAsync()
		{
			var modelPath = await predictionService.TrainAsync<TaxyData, TaxyTripFarePrediction, PoissonRegressor>(_pipelineParameters);
			var result = await predictionService.EvaluateAsync<TaxyData, TaxyTripFarePrediction>(_pipelineTestParameters, modelPath);

			Console.WriteLine($"Rms = {result.Rms}");
			Console.WriteLine($"RSquared = {result.RSquared}");
		}
	}
}
