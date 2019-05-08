using System;
using System.IO;
using System.Reflection;
using System.Threading.Tasks;
using MachineLearningLibrary.Models;
using MachineLearningLibrary.Services;
using Microsoft.ML.Data;
using NUnit.Framework;

namespace MachineLearningLibraryTests
{
	[TestFixture]
	public class MushroomsBinaryClassificationTests
	{
		private PredictionService predictionService = new PredictionService();
		private string _dataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\traindata\mushroom.csv";
		private string _testDataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\testdata\mushroom.csv";
		private char _separator = ',';
		private string _predictedLabel = "PredictedLabel";
		private string[] _dictionarizedLabels = new[] { "Label" };
		private string[] _concatenatedColumns = new[] { "CapShape", "CapSurface", "CapColor", "Bruises", "Odor", "GillAttachment", "GillSpacing", "GillSize", "GillColor", "StalkShape", "StalkRoot", "StalkSurfaceAboveRing", "StalkSurfaceBelowRing", "StalkColorAboveRing", "StalkColorBelowRing", "VeilType", "VeilColor", "RingNumber", "RingType", "SporePrintColor", "Population", "Habitat" };

		[Test]
		public async Task MushroomsBinaryClassificationTest()
		{
			var pipelineParameters = GetPipelineParameters(_dataPath);
			var pipelineTestParameters = GetPipelineParameters(_testDataPath);

			var modelPath = predictionService.Train<MushroomData, MushroomEdiblePrediction>(pipelineParameters, AlgorithmType.FastForestBinaryClassifier);
			var result = predictionService.EvaluateBinaryClassification(pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.FastForestBinaryClassifier), result);

			modelPath = predictionService.Train<MushroomData, MushroomEdiblePrediction>(pipelineParameters, AlgorithmType.AveragedPerceptronBinaryClassifier);
			result = predictionService.EvaluateBinaryClassification(pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.AveragedPerceptronBinaryClassifier), result);

			modelPath = predictionService.Train<MushroomData, MushroomEdiblePrediction>(pipelineParameters, AlgorithmType.AveragedPerceptronBinaryClassifier);
			result = predictionService.EvaluateBinaryClassification(pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.AveragedPerceptronBinaryClassifier), result);

			modelPath = predictionService.Train<MushroomData, MushroomEdiblePrediction>(pipelineParameters, AlgorithmType.FastTreeBinaryClassifier);
			result = predictionService.EvaluateBinaryClassification(pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.FastTreeBinaryClassifier), result);

			modelPath = predictionService.Train<MushroomData, MushroomEdiblePrediction>(pipelineParameters, AlgorithmType.FieldAwareFactorizationMachineBinaryClassifier);
			result = predictionService.EvaluateBinaryClassification(pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.FieldAwareFactorizationMachineBinaryClassifier), result);

			modelPath = predictionService.Train<MushroomData, MushroomEdiblePrediction>(pipelineParameters, AlgorithmType.GeneralizedAdditiveModelBinaryClassifier);
			result = predictionService.EvaluateBinaryClassification(pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.GeneralizedAdditiveModelBinaryClassifier), result);

			modelPath = predictionService.Train<MushroomData, MushroomEdiblePrediction>(pipelineParameters, AlgorithmType.LinearSvmBinaryClassifier);
			result = predictionService.EvaluateBinaryClassification(pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.LinearSvmBinaryClassifier), result);

			modelPath = predictionService.Train<MushroomData, MushroomEdiblePrediction>(pipelineParameters, AlgorithmType.LogisticRegressionBinaryClassifier);
			result = predictionService.EvaluateBinaryClassification(pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.LogisticRegressionBinaryClassifier), result);

			modelPath = predictionService.Train<MushroomData, MushroomEdiblePrediction>(pipelineParameters, AlgorithmType.StochasticDualCoordinateAscentBinaryClassifier);
			result = predictionService.EvaluateBinaryClassification(pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.StochasticDualCoordinateAscentBinaryClassifier), result);

			modelPath = predictionService.Train<MushroomData, MushroomEdiblePrediction>(pipelineParameters, AlgorithmType.StochasticGradientDescentBinaryClassifier);
			result = predictionService.EvaluateBinaryClassification(pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.StochasticGradientDescentBinaryClassifier), result);
		}

		private void LogResult(string algorithm, BinaryClassificationMetrics binaryClassificationMetrics)
		{
			Console.WriteLine($"------------- {algorithm} - EVALUATION RESULTS -------------");
			Console.WriteLine($"Accurancy = {binaryClassificationMetrics.Accuracy}");
			Console.WriteLine($"AUC = {binaryClassificationMetrics.Auc}");
			Console.WriteLine($"F1Score = {binaryClassificationMetrics.F1Score}");
			Console.WriteLine($"------------- {algorithm} - END EVALUATION -------------");
		}

		private PipelineParameters<MushroomData> GetPipelineParameters(string dataPath)
		{
			return new PipelineParameters<MushroomData>(dataPath, _separator, null, _concatenatedColumns);
		}
	}
}
