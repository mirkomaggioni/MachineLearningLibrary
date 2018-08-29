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
	public class IrisDataMulticlassClassificationTests
	{
		private PredictionService predictionService = new PredictionService();
		private PipelineParameters<IrisData> _pipelineParameters;
		private PipelineParameters<IrisData> _pipelineTestParameters;

		[SetUp]
		public void Setup()
		{
			var dir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
			var dataPath = $@"{dir}\traindata\iris.csv";
			var testDataPath = $@"{dir}\testdata\iris.csv";
			var separator = ',';
			var dictionarizedLabels = new[] { "Label" };
			var concatenatedColumns = new[] { "SepalLength", "SepalWidth", "PetalLength", "PetalWidth" };
			_pipelineParameters = new PipelineParameters<IrisData>(dataPath, separator, "PredictedLabel", null, dictionarizedLabels, concatenatedColumns);
			_pipelineTestParameters = new PipelineParameters<IrisData>(testDataPath, separator);
		}

		[Test]
		public async Task IrisDataMulticlassClassificationTest()
		{
			var modelPath = await predictionService.TrainAsync<IrisData, IrisTypePrediction, NaiveBayesClassifier>(_pipelineParameters);
			var result = await predictionService.EvaluateClassificationAsync<IrisData, IrisTypePrediction>(_pipelineTestParameters, modelPath);
			LogResult(nameof(NaiveBayesClassifier), result);

			modelPath = await predictionService.TrainAsync<IrisData, IrisTypePrediction, LogisticRegressionClassifier>(_pipelineParameters);
			result = await predictionService.EvaluateClassificationAsync<IrisData, IrisTypePrediction>(_pipelineTestParameters, modelPath);
			LogResult(nameof(LogisticRegressionClassifier), result);

			modelPath = await predictionService.TrainAsync<IrisData, IrisTypePrediction, StochasticDualCoordinateAscentClassifier>(_pipelineParameters);
			result = await predictionService.EvaluateClassificationAsync<IrisData, IrisTypePrediction>(_pipelineTestParameters, modelPath);
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
