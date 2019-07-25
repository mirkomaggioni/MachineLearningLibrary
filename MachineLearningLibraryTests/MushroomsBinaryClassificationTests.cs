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
		private readonly string _dataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\traindata\mushroom.csv";
		private readonly string _testDataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\testdata\mushroom.csv";
		private readonly char _separator = ',';
		private readonly string _predictedColumn = "Edible";
		private readonly string[] _concatenatedColumns = new[] { "CapShape", "CapSurface", "CapColor", "Bruises", "Odor", "GillAttachment", "GillSpacing", "GillSize", "GillColor", "StalkShape", "StalkRoot", "StalkSurfaceAboveRing", "StalkSurfaceBelowRing", "StalkColorAboveRing", "StalkColorBelowRing", "VeilType", "VeilColor", "RingNumber", "RingType", "SporePrintColor", "Population", "Habitat" };

		[Test]
		public void MushroomsBinaryClassificationTest()
		{
			var pipeline = new Pipeline2<MushroomData>(_dataPath, _separator, AlgorithmType.FastForestBinaryClassifier, (_predictedColumn, false, DataKind.Boolean), _concatenatedColumns, _concatenatedColumns);
			var pipelineTest = new Pipeline2<MushroomData>(_testDataPath, _separator);
			pipeline.BuildModel();

			var result = pipeline.EvaluateBinaryClassification(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.FastForestBinaryClassifier), result);

			pipeline = new Pipeline2<MushroomData>(_dataPath, _separator, AlgorithmType.AveragedPerceptronBinaryClassifier, (_predictedColumn, false, DataKind.Boolean), _concatenatedColumns, _concatenatedColumns);
			pipeline.BuildModel();
			result = pipeline.EvaluateBinaryClassification(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.AveragedPerceptronBinaryClassifier), result);

			pipeline = new Pipeline2<MushroomData>(_dataPath, _separator, AlgorithmType.FastTreeBinaryClassifier, (_predictedColumn, false, DataKind.Boolean), _concatenatedColumns, _concatenatedColumns);
			pipeline.BuildModel();
			result = pipeline.EvaluateBinaryClassification(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.FastTreeBinaryClassifier), result);

			pipeline = new Pipeline2<MushroomData>(_dataPath, _separator, AlgorithmType.FieldAwareFactorizationMachineBinaryClassifier, (_predictedColumn, false, DataKind.Boolean), _concatenatedColumns, _concatenatedColumns);
			pipeline.BuildModel();
			result = pipeline.EvaluateBinaryClassification(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.FieldAwareFactorizationMachineBinaryClassifier), result);

			pipeline = new Pipeline2<MushroomData>(_dataPath, _separator, AlgorithmType.GamBinaryClassifier, (_predictedColumn, false, DataKind.Boolean), _concatenatedColumns, _concatenatedColumns);
			pipeline.BuildModel();
			result = pipeline.EvaluateBinaryClassification(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.GamBinaryClassifier), result);

			pipeline = new Pipeline2<MushroomData>(_dataPath, _separator, AlgorithmType.LinearSvmBinaryClassifier, (_predictedColumn, false, DataKind.Boolean), _concatenatedColumns, _concatenatedColumns);
			pipeline.BuildModel();
			result = pipeline.EvaluateBinaryClassification(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.LinearSvmBinaryClassifier), result);

			pipeline = new Pipeline2<MushroomData>(_dataPath, _separator, AlgorithmType.LbfgsBinaryClassifier, (_predictedColumn, false, DataKind.Boolean), _concatenatedColumns, _concatenatedColumns);
			pipeline.BuildModel();
			result = pipeline.EvaluateBinaryClassification(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.LbfgsBinaryClassifier), result);

			pipeline = new Pipeline2<MushroomData>(_dataPath, _separator, AlgorithmType.StochasticDualCoordinateAscentBinaryClassifier, (_predictedColumn, false, DataKind.Boolean), _concatenatedColumns, _concatenatedColumns);
			pipeline.BuildModel();
			result = pipeline.EvaluateBinaryClassification(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.StochasticDualCoordinateAscentBinaryClassifier), result);

			pipeline = new Pipeline2<MushroomData>(_dataPath, _separator, AlgorithmType.StochasticGradientDescentBinaryClassifier, (_predictedColumn, false, DataKind.Boolean), _concatenatedColumns, _concatenatedColumns);
			pipeline.BuildModel();
			result = pipeline.EvaluateBinaryClassification(pipelineTest.DataView);
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
	}
}
