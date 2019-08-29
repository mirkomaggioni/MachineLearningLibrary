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
	public class GlassDataClusteringTests
	{
		private readonly string _dataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\traindata\glass.csv";
		private readonly string _testDataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\testdata\glass.csv";
		private readonly char _separator = ',';
		private readonly string _predictedColumn = "Type";
		private readonly string[] _concatenatedColumns = new[] { "IdNumber", "RefractiveIndex", "Sodium", "Magnesium", "Aluminium", "Silicon", "Potassium", "Calcium", "Barium", "Iron" };

		[Test]
		public void GlassDataClusteringTest()
		{
			var pipeline = new Pipeline<Glass>(_dataPath, _separator, AlgorithmType.NaiveBayesMultiClassifier, new PredictedColumn(_predictedColumn, true), _concatenatedColumns);
			var pipelineTest = new Pipeline<Glass>(_testDataPath, _separator);
			pipeline.BuildPipeline();
			pipeline.BuildModel();

			var result = pipeline.EvaluateClustering(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.NaiveBayesMultiClassifier), result);

			pipeline = new Pipeline<Glass>(_dataPath, _separator, AlgorithmType.LbfgsMultiClassifier, new PredictedColumn(_predictedColumn, true), _concatenatedColumns);
			pipeline.BuildPipeline();
			pipeline.BuildModel();
			result = pipeline.EvaluateClustering(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.LbfgsMultiClassifier), result);

			pipeline = new Pipeline<Glass>(_dataPath, _separator, AlgorithmType.StochasticDualCoordinateAscentMultiClassifier, new PredictedColumn(_predictedColumn, true), _concatenatedColumns);
			pipeline.BuildPipeline();
			pipeline.BuildModel();
			result = pipeline.EvaluateClustering(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.StochasticDualCoordinateAscentMultiClassifier), result);
		}

		[Test]
		[TestCase(54f, 1.51837f, 13.14f, 2.84f, 1.28f, 72.85f, 0.55f, 9.07f, 0.00f, 0.00f, 1u)]
		[TestCase(110f, 1.51818f, 13.72f, 0.00f, 0.56f, 74.45f, 0.00f, 10.99f, 0.00f, 0.00f, 2u)]
		[TestCase(187f, 1.51838f, 14.32f, 3.26f, 2.22f, 71.25f, 1.46f, 5.79f, 1.63f, 0.00f, 7u)]
		public void GlassDataClusteringPredictTest(float idNumber, float refractiveIndex, float sodium, float magnesium, float aluminium, float silicon, float potassium, float calcium, float barium, float iron, uint type)
		{
			var pipeline = new Pipeline<Glass>(_dataPath, _separator, AlgorithmType.StochasticDualCoordinateAscentMultiClassifier, new PredictedColumn(_predictedColumn, true), _concatenatedColumns);
			pipeline.BuildPipeline();
			pipeline.BuildModel();

			var prediction = pipeline.PredictScore<Glass, GlassTypePrediction, uint>(new Glass() { IdNumber = idNumber, RefractiveIndex = refractiveIndex, Sodium = sodium, Magnesium = magnesium, Aluminium = aluminium, Silicon = silicon, Potassium = potassium, Calcium = calcium,Barium = barium, Iron = iron });

			Assert.AreEqual(prediction.PredictedLabel, type);
		}

		private void LogResult(string algorithm, ClusteringMetrics clusteringMetrics)
		{
			Console.WriteLine($"------------- {algorithm} - EVALUATION RESULTS -------------");
			Console.WriteLine($"AVG MIN SCORE = {clusteringMetrics.AverageDistance}");
			Console.WriteLine($"DBI = {clusteringMetrics.DaviesBouldinIndex}");
			Console.WriteLine($"NMI = {clusteringMetrics.NormalizedMutualInformation}");
			Console.WriteLine($"------------- {algorithm} - END EVALUATION -------------");
		}
	}
}
