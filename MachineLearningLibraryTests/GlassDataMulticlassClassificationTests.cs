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
	public class GlassDataMulticlassClassificationTests
	{
		private PredictionService predictionService = new PredictionService();
		private string _dataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\traindata\glass.csv";
		private string _testDataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\testdata\glass.csv";
		private char _separator = ',';
		private string[] _dictionarizedLabels = new[] { "Label" };
		private string[] _concatenatedColumns = new[] { "IdNumber", "RefractiveIndex", "Sodium", "Magnesium", "Aluminium", "Silicon", "Potassium", "Calcium", "Barium", "Iron" };
		private string _predictedLabel = "PredictedLabel";

		[Test]
		public async Task GlassDataMulticlassClassificationTest()
		{
			var pipelineParameters = GetPipelineParameters(new NaiveBayesClassifier());
			var pipelineTestParameters = GetPipelineTestParameters();
			var modelPath = await predictionService.TrainAsync<GlassData, GlassTypePrediction>(pipelineParameters);
			var result = await predictionService.EvaluateClassificationAsync<GlassData, GlassTypePrediction>(pipelineTestParameters, modelPath);
			LogResult(nameof(NaiveBayesClassifier), result);

			pipelineParameters = GetPipelineParameters(new LogisticRegressionClassifier());
			modelPath = await predictionService.TrainAsync<GlassData, GlassTypePrediction>(pipelineParameters);
			result = await predictionService.EvaluateClassificationAsync<GlassData, GlassTypePrediction>(pipelineTestParameters, modelPath);
			LogResult(nameof(LogisticRegressionClassifier), result);

			pipelineParameters = GetPipelineParameters(new StochasticDualCoordinateAscentClassifier());
			modelPath = await predictionService.TrainAsync<GlassData, GlassTypePrediction>(pipelineParameters);
			result = await predictionService.EvaluateClassificationAsync<GlassData, GlassTypePrediction>(pipelineTestParameters, modelPath);
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

		private PipelineParameters<GlassData> GetPipelineParameters(ILearningPipelineItem algorithm) {
			return new PipelineParameters<GlassData>(_dataPath, _separator, _predictedLabel, null, _dictionarizedLabels, _concatenatedColumns, algorithm);
		}

		private PipelineParameters<GlassData> GetPipelineTestParameters() {
			return new PipelineParameters<GlassData>(_testDataPath, _separator);
		}
	}
}
