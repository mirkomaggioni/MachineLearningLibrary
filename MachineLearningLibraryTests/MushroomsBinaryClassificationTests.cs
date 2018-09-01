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
	public class MushroomsBinaryClassificationTests
	{
		private PredictionService predictionService = new PredictionService();
		private PipelineParameters<MushroomData> _pipelineParameters;
		private PipelineParameters<MushroomData> _pipelineTestParameters;

		[SetUp]
		public void Setup()
		{
			var dir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
			var dataPath = $@"{dir}\traindata\mushroom.csv";
			var testDataPath = $@"{dir}\testdata\mushroom.csv";
			var separator = ',';
			var dictionarizedLabels = new[] { "Label" };
			var concatenatedColumns = new[] { "CapShape", "CapSurface", "CapColor", "Bruises", "Odor", "GillAttachment", "GillSpacing", "GillSize", "GillColor", "StalkShape", "StalkRoot", "StalkSurfaceAboveRing", "StalkSurfaceBelowRing", "StalkColorAboveRing", "StalkColorBelowRing", "VeilType", "VeilColor", "RingNumber", "RingType", "SporePrintColor", "Population", "Habitat" };
			_pipelineParameters = new PipelineParameters<MushroomData>(dataPath, separator, "PredictedLabel", concatenatedColumns, dictionarizedLabels, concatenatedColumns);
			_pipelineTestParameters = new PipelineParameters<MushroomData>(testDataPath, separator);
		}

		[Test]
		public async Task MushroomsBinaryClassificationTest()
		{
			var modelPath = await predictionService.TrainAsync<MushroomData, MushroomEdiblePrediction, AveragedPerceptronBinaryClassifier>(_pipelineParameters);
			var result = await predictionService.EvaluateBinaryClassificationAsync<MushroomData, MushroomEdiblePrediction>(_pipelineTestParameters, modelPath);
			LogResult(nameof(AveragedPerceptronBinaryClassifier), result);

			modelPath = await predictionService.TrainAsync<MushroomData, MushroomEdiblePrediction, FastForestBinaryClassifier>(_pipelineParameters);
			result = await predictionService.EvaluateBinaryClassificationAsync<MushroomData, MushroomEdiblePrediction>(_pipelineTestParameters, modelPath);
			LogResult(nameof(FastForestBinaryClassifier), result);

			modelPath = await predictionService.TrainAsync<MushroomData, MushroomEdiblePrediction, FastTreeBinaryClassifier>(_pipelineParameters);
			result = await predictionService.EvaluateBinaryClassificationAsync<MushroomData, MushroomEdiblePrediction>(_pipelineTestParameters, modelPath);
			LogResult(nameof(FastTreeBinaryClassifier), result);

			modelPath = await predictionService.TrainAsync<MushroomData, MushroomEdiblePrediction, FieldAwareFactorizationMachineBinaryClassifier>(_pipelineParameters);
			result = await predictionService.EvaluateBinaryClassificationAsync<MushroomData, MushroomEdiblePrediction>(_pipelineTestParameters, modelPath);
			LogResult(nameof(FieldAwareFactorizationMachineBinaryClassifier), result);

			modelPath = await predictionService.TrainAsync<MushroomData, MushroomEdiblePrediction, GeneralizedAdditiveModelBinaryClassifier>(_pipelineParameters);
			result = await predictionService.EvaluateBinaryClassificationAsync<MushroomData, MushroomEdiblePrediction>(_pipelineTestParameters, modelPath);
			LogResult(nameof(GeneralizedAdditiveModelBinaryClassifier), result);

			modelPath = await predictionService.TrainAsync<MushroomData, MushroomEdiblePrediction, LinearSvmBinaryClassifier>(_pipelineParameters);
			result = await predictionService.EvaluateBinaryClassificationAsync<MushroomData, MushroomEdiblePrediction>(_pipelineTestParameters, modelPath);
			LogResult(nameof(LinearSvmBinaryClassifier), result);

			modelPath = await predictionService.TrainAsync<MushroomData, MushroomEdiblePrediction, LogisticRegressionBinaryClassifier>(_pipelineParameters);
			result = await predictionService.EvaluateBinaryClassificationAsync<MushroomData, MushroomEdiblePrediction>(_pipelineTestParameters, modelPath);
			LogResult(nameof(LogisticRegressionBinaryClassifier), result);

			modelPath = await predictionService.TrainAsync<MushroomData, MushroomEdiblePrediction, StochasticDualCoordinateAscentBinaryClassifier>(_pipelineParameters);
			result = await predictionService.EvaluateBinaryClassificationAsync<MushroomData, MushroomEdiblePrediction>(_pipelineTestParameters, modelPath);
			LogResult(nameof(StochasticDualCoordinateAscentBinaryClassifier), result);

			modelPath = await predictionService.TrainAsync<MushroomData, MushroomEdiblePrediction, StochasticGradientDescentBinaryClassifier>(_pipelineParameters);
			result = await predictionService.EvaluateBinaryClassificationAsync<MushroomData, MushroomEdiblePrediction>(_pipelineTestParameters, modelPath);
			LogResult(nameof(StochasticGradientDescentBinaryClassifier), result);
		}

		private void LogResult(string algorithm, BinaryClassificationMetrics classificationMetrics)
		{
			Console.WriteLine($"------------- {algorithm} - EVALUATION RESULTS -------------");
			Console.WriteLine($"Accuracy = {classificationMetrics.Accuracy}");
			Console.WriteLine($"Auc = {classificationMetrics.Auc}");
			Console.WriteLine($"F1Score = {classificationMetrics.F1Score}");
			Console.WriteLine($"LogLoss = {classificationMetrics.LogLoss}");
			Console.WriteLine($"LogLossReduction = {classificationMetrics.LogLossReduction}");
			Console.WriteLine($"------------- {algorithm} - END EVALUATION -------------");
		}
	}
}
