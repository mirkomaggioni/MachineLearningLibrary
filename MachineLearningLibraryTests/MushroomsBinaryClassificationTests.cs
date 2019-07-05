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
			var pipelineTestParameters = new Pipeline<MushroomData>(_testDataPath, _separator);

			var pipelineTransformer = pipelineParameters.Train(AlgorithmType.FastForestBinaryClassifier);
			var result = pipelineTransformer.EvaluateBinaryClassification(pipelineTestParameters.DataView);
			LogResult(nameof(AlgorithmType.FastForestBinaryClassifier), result);

			pipelineTransformer = pipelineParameters.Train(AlgorithmType.AveragedPerceptronBinaryClassifier);
			result = pipelineTransformer.EvaluateBinaryClassification(pipelineTestParameters.DataView);
			LogResult(nameof(AlgorithmType.AveragedPerceptronBinaryClassifier), result);

			pipelineTransformer = pipelineParameters.Train(AlgorithmType.FastTreeBinaryClassifier);
			result = pipelineTransformer.EvaluateBinaryClassification(pipelineTestParameters.DataView);
			LogResult(nameof(AlgorithmType.FastTreeBinaryClassifier), result);

			pipelineTransformer = pipelineParameters.Train(AlgorithmType.FieldAwareFactorizationMachineBinaryClassifier);
			result = pipelineTransformer.EvaluateBinaryClassification(pipelineTestParameters.DataView);
			LogResult(nameof(AlgorithmType.FieldAwareFactorizationMachineBinaryClassifier), result);

			pipelineTransformer = pipelineParameters.Train(AlgorithmType.GamBinaryClassifier);
			result = pipelineTransformer.EvaluateBinaryClassification(pipelineTestParameters.DataView);
			LogResult(nameof(AlgorithmType.GamBinaryClassifier), result);

			pipelineTransformer = pipelineParameters.Train(AlgorithmType.LinearSvmBinaryClassifier);
			result = pipelineTransformer.EvaluateBinaryClassification(pipelineTestParameters.DataView);
			LogResult(nameof(AlgorithmType.LinearSvmBinaryClassifier), result);

			pipelineTransformer = pipelineParameters.Train(AlgorithmType.LbfgsBinaryClassifier);
			result = pipelineTransformer.EvaluateBinaryClassification(pipelineTestParameters.DataView);
			LogResult(nameof(AlgorithmType.LbfgsBinaryClassifier), result);

			pipelineTransformer = pipelineParameters.Train(AlgorithmType.StochasticDualCoordinateAscentBinaryClassifier);
			result = pipelineTransformer.EvaluateBinaryClassification(pipelineTestParameters.DataView);
			LogResult(nameof(AlgorithmType.StochasticDualCoordinateAscentBinaryClassifier), result);

			pipelineTransformer = pipelineParameters.Train(AlgorithmType.StochasticGradientDescentBinaryClassifier);
			result = pipelineTransformer.EvaluateBinaryClassification(pipelineTestParameters.DataView);
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

		private ITrain GetPipelineParameters(string dataPath)
		{
			return (new Pipeline<CarData>(dataPath, _separator))
				.CopyColumn("Label", "Edible")
				.ConcatenateColumns(_concatenatedColumns);
		}
	}
}
