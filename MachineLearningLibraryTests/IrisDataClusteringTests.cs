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
	public class IrisDataClusteringTests
	{
		private readonly string _dataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\traindata\iris.csv";
		private readonly string _testDataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\testdata\iris.csv";
		private readonly char _separator = ',';
		private readonly string _predictedColumn = "Type";
		private readonly string[] _concatenatedColumns = new[] { "SepalLength", "SepalWidth", "PetalLength", "PetalWidth" };

		[Test]
		public void IrisDataClusteringTest()
		{
			var pipeline = new Pipeline<Iris>(_dataPath, _separator, AlgorithmType.NaiveBayesMultiClassifier, new PredictedColumn(_predictedColumn, true), _concatenatedColumns);
			var pipelineTest = new Pipeline<Iris>(_testDataPath, _separator);
			pipeline.BuildModel();

			var result = pipeline.EvaluateClustering(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.NaiveBayesMultiClassifier), result);

			pipeline = new Pipeline<Iris>(_dataPath, _separator, AlgorithmType.LbfgsMultiClassifier, new PredictedColumn(_predictedColumn, true), _concatenatedColumns);
			pipeline.BuildModel();
			result = pipeline.EvaluateClustering(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.LbfgsMultiClassifier), result);

			pipeline = new Pipeline<Iris>(_dataPath, _separator, AlgorithmType.StochasticDualCoordinateAscentMultiClassifier, new PredictedColumn(_predictedColumn, true), _concatenatedColumns);
			pipeline.BuildModel();
			result = pipeline.EvaluateClustering(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.StochasticDualCoordinateAscentMultiClassifier), result);
		}

		[Test]
		[TestCase(6.1f, 3.0f, 4.6f, 1.4f, "Iris-versicolor")]
		[TestCase(5.5f, 3.5f, 1.3f, 0.2f, "Iris-setosa")]
		[TestCase(5.0f, 2.3f, 3.3f, 1.0f, "Iris-versicolor")]
		[TestCase(7.9f, 3.8f, 6.4f, 2.0f, "Iris-virginica")]
		public void IrisDataClusteringPredictTest(float sepalLength, float sepalWidth, float petalLength, float petalWidth, string type)
		{
			var pipeline = new Pipeline<Iris>(_dataPath, _separator, AlgorithmType.NaiveBayesMultiClassifier, new PredictedColumn(_predictedColumn, true), _concatenatedColumns);
			pipeline.BuildModel();

			var prediction = pipeline.PredictScore<Iris, IrisTypePrediction, string>(new Iris() { SepalLength = sepalLength, SepalWidth = sepalWidth, PetalLength = petalLength, PetalWidth = petalWidth });

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
