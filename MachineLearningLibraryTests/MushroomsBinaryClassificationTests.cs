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
	public class MushroomsBinaryClassificationTests
	{
		private readonly PredictionService predictionService = new PredictionService();
		private readonly string _dataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\traindata\mushroom.csv";
		private readonly string _testDataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\testdata\mushroom.csv";
		private readonly char _separator = ',';
		private readonly string _predictedColumn = "Edible";
		private readonly string[] _concatenatedColumns = new[] { "CapShape", "CapSurface", "CapColor", "Bruises", "Odor", "GillAttachment", "GillSpacing", "GillSize", "GillColor", "StalkShape", "StalkRoot", "StalkSurfaceAboveRing", "StalkSurfaceBelowRing", "StalkColorAboveRing", "StalkColorBelowRing", "VeilType", "VeilColor", "RingNumber", "RingType", "SporePrintColor", "Population", "Habitat" };

		[Test]
		public void MushroomsBinaryClassificationTest()
		{
			var pipelineParameters = GetPipelineParameters(_dataPath);
			var pipelineTestParameters = GetPipelineParameters(_testDataPath);

			var model = predictionService.Train<MushroomData, MushroomEdiblePrediction>(pipelineParameters, AlgorithmType.FastForestBinaryClassifier);
			var result = predictionService.EvaluateBinaryClassification(model, pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.FastForestBinaryClassifier), result);

			model = predictionService.Train<MushroomData, MushroomEdiblePrediction>(pipelineParameters, AlgorithmType.AveragedPerceptronBinaryClassifier);
			result = predictionService.EvaluateBinaryClassification(model, pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.AveragedPerceptronBinaryClassifier), result);

			model = predictionService.Train<MushroomData, MushroomEdiblePrediction>(pipelineParameters, AlgorithmType.AveragedPerceptronBinaryClassifier);
			result = predictionService.EvaluateBinaryClassification(model, pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.AveragedPerceptronBinaryClassifier), result);

			model = predictionService.Train<MushroomData, MushroomEdiblePrediction>(pipelineParameters, AlgorithmType.FastTreeBinaryClassifier);
			result = predictionService.EvaluateBinaryClassification(model, pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.FastTreeBinaryClassifier), result);

			model = predictionService.Train<MushroomData, MushroomEdiblePrediction>(pipelineParameters, AlgorithmType.FieldAwareFactorizationMachineBinaryClassifier);
			result = predictionService.EvaluateBinaryClassification(model, pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.FieldAwareFactorizationMachineBinaryClassifier), result);

			model = predictionService.Train<MushroomData, MushroomEdiblePrediction>(pipelineParameters, AlgorithmType.GamBinaryClassifier);
			result = predictionService.EvaluateBinaryClassification(model, pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.GamBinaryClassifier), result);

			model = predictionService.Train<MushroomData, MushroomEdiblePrediction>(pipelineParameters, AlgorithmType.LinearSvmBinaryClassifier);
			result = predictionService.EvaluateBinaryClassification(model, pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.LinearSvmBinaryClassifier), result);

			model = predictionService.Train<MushroomData, MushroomEdiblePrediction>(pipelineParameters, AlgorithmType.LbfgsBinaryClassifier);
			result = predictionService.EvaluateBinaryClassification(model, pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.LbfgsBinaryClassifier), result);

			model = predictionService.Train<MushroomData, MushroomEdiblePrediction>(pipelineParameters, AlgorithmType.StochasticDualCoordinateAscentBinaryClassifier);
			result = predictionService.EvaluateBinaryClassification(model, pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.StochasticDualCoordinateAscentBinaryClassifier), result);

			model = predictionService.Train<MushroomData, MushroomEdiblePrediction>(pipelineParameters, AlgorithmType.StochasticGradientDescentBinaryClassifier);
			result = predictionService.EvaluateBinaryClassification(model, pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.StochasticGradientDescentBinaryClassifier), result);
		}

		private void LogResult(string algorithm, BinaryClassificationMetrics binaryClassificationMetrics)
		{
			Console.WriteLine($"------------- {algorithm} - EVALUATION RESULTS -------------");
			Console.WriteLine($"Accurancy = {binaryClassificationMetrics.Accuracy}");
			Console.WriteLine($"AUC = {binaryClassificationMetrics.AreaUnderRocCurve}");
			Console.WriteLine($"F1Score = {binaryClassificationMetrics.F1Score}");
			Console.WriteLine($"------------- {algorithm} - END EVALUATION -------------");
		}

		private Pipeline<MushroomData> GetPipelineParameters(string dataPath)
		{
			return new Pipeline<MushroomData>(dataPath, _separator);
		}
	}
}
