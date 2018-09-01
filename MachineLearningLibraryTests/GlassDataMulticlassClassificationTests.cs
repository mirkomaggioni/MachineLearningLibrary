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
	public class GlassDataMulticlassClassificationTests
	{
		private PredictionService predictionService = new PredictionService();
		private PipelineParameters<GlassData> _pipelineParameters;
		private PipelineParameters<GlassData> _pipelineTestParameters;

		[SetUp]
		public void Setup()
		{
			var dir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
			var dataPath = $@"{dir}\traindata\glass.csv";
			var testDataPath = $@"{dir}\testdata\glass.csv";
			var separator = ',';
			var dictionarizedLabels = new[] { "Label" };
			var concatenatedColumns = new[] { "IdNumber", "RefractiveIndex", "Sodium", "Magnesium", "Aluminium", "Silicon", "Potassium", "Calcium", "Barium", "Iron" };
			_pipelineParameters = new PipelineParameters<GlassData>(dataPath, separator, "PredictedLabel", null, dictionarizedLabels, concatenatedColumns);
			_pipelineTestParameters = new PipelineParameters<GlassData>(testDataPath, separator);
		}

		[Test]
		public async Task GlassDataMulticlassClassificationTest()
		{
			var modelPath = await predictionService.TrainAsync<GlassData, GlassTypePrediction, NaiveBayesClassifier>(_pipelineParameters);
			var result = await predictionService.EvaluateClassificationAsync<GlassData, GlassTypePrediction>(_pipelineTestParameters, modelPath);
			LogResult(nameof(NaiveBayesClassifier), result);

			modelPath = await predictionService.TrainAsync<GlassData, GlassTypePrediction, LogisticRegressionClassifier>(_pipelineParameters);
			result = await predictionService.EvaluateClassificationAsync<GlassData, GlassTypePrediction>(_pipelineTestParameters, modelPath);
			LogResult(nameof(LogisticRegressionClassifier), result);

			modelPath = await predictionService.TrainAsync<GlassData, GlassTypePrediction, StochasticDualCoordinateAscentClassifier>(_pipelineParameters);
			result = await predictionService.EvaluateClassificationAsync<GlassData, GlassTypePrediction>(_pipelineTestParameters, modelPath);
			LogResult(nameof(StochasticDualCoordinateAscentClassifier), result);
		}

		private void LogResult(string algorithm, ClassificationMetrics classificationMetrics)
		{
			Console.WriteLine($"------------- {algorithm} - EVALUATION RESULTS -------------");
			Console.WriteLine($"AccuracyMacro = {classificationMetrics.AccuracyMacro}");
			Console.WriteLine($"AccuracyMicro = {classificationMetrics.AccuracyMicro}");
			Console.WriteLine($"LogLoss = {classificationMetrics.LogLoss}");
			Console.WriteLine($"LogLossReduction = {classificationMetrics.LogLossReduction}");
			Console.WriteLine($"------------- {algorithm} - END EVALUATION -------------");
		}
	}
}
