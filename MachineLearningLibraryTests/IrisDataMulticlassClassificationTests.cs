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
	public class IrisDataMulticlassClassificationTests
	{
		private PredictionService predictionService = new PredictionService();
		private string _dataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\traindata\iris.csv";
		private string _testDataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\testdata\iris.csv";
		private char _separator = ',';
		private string _predictedLabel = "PredictedLabel";
		private string[] _dictionarizedLabels = new[] { "Label" };
		private string[] _concatenatedColumns = new[] { "SepalLength", "SepalWidth", "PetalLength", "PetalWidth" };

		[Test]
		public async Task IrisDataMulticlassClassificationTest()
		{
			var pipelineParameters = GetPipelineParameters(new NaiveBayesClassifier());
			var pipelineTestParameters = GetPipelineTestParameters();
			var modelPath = await predictionService.TrainAsync<IrisData, IrisTypePrediction>(pipelineParameters);
			var result = await predictionService.EvaluateClassificationAsync<IrisData, IrisTypePrediction>(pipelineTestParameters, modelPath);
			LogResult(nameof(NaiveBayesClassifier), result);

			pipelineParameters = GetPipelineParameters(new LogisticRegressionClassifier());
			modelPath = await predictionService.TrainAsync<IrisData, IrisTypePrediction>(pipelineParameters);
			result = await predictionService.EvaluateClassificationAsync<IrisData, IrisTypePrediction>(pipelineTestParameters, modelPath);
			LogResult(nameof(LogisticRegressionClassifier), result);

			pipelineParameters = GetPipelineParameters(new StochasticDualCoordinateAscentClassifier());
			modelPath = await predictionService.TrainAsync<IrisData, IrisTypePrediction>(pipelineParameters);
			result = await predictionService.EvaluateClassificationAsync<IrisData, IrisTypePrediction>(pipelineTestParameters, modelPath);
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

		private PipelineParameters<IrisData> GetPipelineParameters(ILearningPipelineItem algorithm) {
			return new PipelineParameters<IrisData>(_dataPath, _separator, _predictedLabel, null, _dictionarizedLabels, _concatenatedColumns, algorithm);
		}

		private PipelineParameters<IrisData> GetPipelineTestParameters() {
			return new PipelineParameters<IrisData>(_testDataPath, _separator);
		}
	}
}
