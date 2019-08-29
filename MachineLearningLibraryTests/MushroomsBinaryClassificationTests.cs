using System;
using System.IO;
using System.Reflection;
using MachineLearningLibrary.Models;
using MachineLearningLibrary.Models.Data;
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
			var pipeline = new Pipeline<Mushroom>(_dataPath, _separator, AlgorithmType.FastForestBinaryClassifier, new PredictedColumn(_predictedColumn, dataKind: DataKind.Boolean), _concatenatedColumns, _concatenatedColumns);
			var pipelineTest = new Pipeline<Mushroom>(_testDataPath, _separator);
			pipeline.BuildPipeline();
			pipeline.BuildModel();

			var result = pipeline.EvaluateBinaryClassification(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.FastForestBinaryClassifier), result);

			pipeline = new Pipeline<Mushroom>(_dataPath, _separator, AlgorithmType.AveragedPerceptronBinaryClassifier, new PredictedColumn(_predictedColumn, dataKind: DataKind.Boolean), _concatenatedColumns, _concatenatedColumns);
			pipeline.BuildPipeline();
			pipeline.BuildModel();
			result = pipeline.EvaluateBinaryClassification(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.AveragedPerceptronBinaryClassifier), result);

			pipeline = new Pipeline<Mushroom>(_dataPath, _separator, AlgorithmType.FastTreeBinaryClassifier, new PredictedColumn(_predictedColumn, dataKind: DataKind.Boolean), _concatenatedColumns, _concatenatedColumns);
			pipeline.BuildPipeline();
			pipeline.BuildModel();
			result = pipeline.EvaluateBinaryClassification(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.FastTreeBinaryClassifier), result);

			pipeline = new Pipeline<Mushroom>(_dataPath, _separator, AlgorithmType.FieldAwareFactorizationMachineBinaryClassifier, new PredictedColumn(_predictedColumn, dataKind: DataKind.Boolean), _concatenatedColumns, _concatenatedColumns);
			pipeline.BuildPipeline();
			pipeline.BuildModel();
			result = pipeline.EvaluateBinaryClassification(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.FieldAwareFactorizationMachineBinaryClassifier), result);

			pipeline = new Pipeline<Mushroom>(_dataPath, _separator, AlgorithmType.GamBinaryClassifier, new PredictedColumn(_predictedColumn, dataKind: DataKind.Boolean), _concatenatedColumns, _concatenatedColumns);
			pipeline.BuildPipeline();
			pipeline.BuildModel();
			result = pipeline.EvaluateBinaryClassification(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.GamBinaryClassifier), result);

			pipeline = new Pipeline<Mushroom>(_dataPath, _separator, AlgorithmType.LinearSvmBinaryClassifier, new PredictedColumn(_predictedColumn, dataKind: DataKind.Boolean), _concatenatedColumns, _concatenatedColumns);
			pipeline.BuildPipeline();
			pipeline.BuildModel();
			result = pipeline.EvaluateBinaryClassification(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.LinearSvmBinaryClassifier), result);

			pipeline = new Pipeline<Mushroom>(_dataPath, _separator, AlgorithmType.LbfgsBinaryClassifier, new PredictedColumn(_predictedColumn, dataKind: DataKind.Boolean), _concatenatedColumns, _concatenatedColumns);
			pipeline.BuildPipeline();
			pipeline.BuildModel();
			result = pipeline.EvaluateBinaryClassification(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.LbfgsBinaryClassifier), result);

			pipeline = new Pipeline<Mushroom>(_dataPath, _separator, AlgorithmType.StochasticDualCoordinateAscentBinaryClassifier, new PredictedColumn(_predictedColumn, dataKind: DataKind.Boolean), _concatenatedColumns, _concatenatedColumns);
			pipeline.BuildPipeline();
			pipeline.BuildModel();
			result = pipeline.EvaluateBinaryClassification(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.StochasticDualCoordinateAscentBinaryClassifier), result);

			pipeline = new Pipeline<Mushroom>(_dataPath, _separator, AlgorithmType.StochasticGradientDescentBinaryClassifier, new PredictedColumn(_predictedColumn, dataKind: DataKind.Boolean), _concatenatedColumns, _concatenatedColumns);
			pipeline.BuildPipeline();
			pipeline.BuildModel();
			result = pipeline.EvaluateBinaryClassification(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.StochasticGradientDescentBinaryClassifier), result);
		}

		[Test]
		[TestCase("x", "s", "n", "t", "p", "f", "c", "n", "n", "e", "e", "s", "s", "w", "w", "p", "w", "o", "p", "n", "v", "u", false)]
		[TestCase("k", "y", "n", "f", "y", "f", "c", "n", "b", "t", "?", "s", "k", "w", "w", "p", "w", "o", "e", "w", "v", "d", false)]
		[TestCase("k", "s", "n", "f", "f", "f", "c", "n", "b", "t", "?", "k", "s", "w", "p", "p", "w", "o", "e", "w", "v", "l", false)]
		[TestCase("b", "s", "g", "f", "n", "f", "w", "b", "g", "e", "?", "s", "k", "w", "w", "p", "w", "t", "p", "w", "n", "g", true)]
		public void TaxyDataRegressionPredictTest(string capShape, string capSurface, string capColor, string bruises, string odor, string gillAttachment, string gillSpacing, string gillSize, string gillColor, string stalkShape, string stalkRoot, string stalkSurfaceAboveRing, string stalkSurfaceBelowRing, string stalkColorAboveRing, string stalkColorBelowRing, string veilType, string veilColor, string ringNumber, string ringType, string sporePrintColor, string population, string habitat, bool edible)
		{
			var pipeline = new Pipeline<Mushroom>(_dataPath, _separator, AlgorithmType.FastForestBinaryClassifier, new PredictedColumn(_predictedColumn, dataKind: DataKind.Boolean), _concatenatedColumns, _concatenatedColumns);
			pipeline.BuildPipeline();
			pipeline.BuildModel();

			var prediction = pipeline.PredictScore<Mushroom, MushroomEdiblePrediction, bool>(new Mushroom() { CapShape = capShape, CapSurface = capSurface, CapColor = capColor, Bruises = bruises, Odor = odor, GillAttachment = gillAttachment, GillSpacing = gillSpacing, GillSize = gillSize, GillColor = gillColor, StalkShape = stalkShape, StalkRoot = stalkRoot, StalkSurfaceAboveRing = stalkSurfaceAboveRing, StalkSurfaceBelowRing = stalkSurfaceBelowRing, StalkColorAboveRing = stalkColorAboveRing, StalkColorBelowRing = stalkColorBelowRing, VeilType = veilType, VeilColor = veilColor, RingNumber = ringNumber, RingType = ringType, SporePrintColor = sporePrintColor, Population = population, Habitat = habitat });
			Assert.That(prediction.PredictedLabel == edible);
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
