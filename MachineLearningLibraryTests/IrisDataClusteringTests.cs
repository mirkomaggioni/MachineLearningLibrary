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
	public class IrisDataClusteringTests
	{
		private readonly PredictionService predictionService = new PredictionService();
		private readonly string _dataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\traindata\iris.csv";
		private readonly string _testDataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\testdata\iris.csv";
		private readonly char _separator = ',';
		private readonly string _predictedColumn = "Type";
		private readonly string[] _concatenatedColumns = new[] { "SepalLength", "SepalWidth", "PetalLength", "PetalWidth" };

		[Test]
		public void IrisDataClusteringTest()
		{
			var pipelineParameters = GetPipelineParameters(_dataPath);
			var pipelineTestParameters = new Pipeline<IrisData>(_testDataPath, _separator);

			var pipelineTransformer = pipelineParameters.Train(AlgorithmType.NaiveBayesMultiClassifier);
			var result = pipelineTransformer.EvaluateClustering(pipelineTestParameters.DataView);
			LogResult(nameof(AlgorithmType.NaiveBayesMultiClassifier), result);

			pipelineTransformer = pipelineParameters.Train(AlgorithmType.LbfgsMultiClassifier);
			result = pipelineTransformer.EvaluateClustering(pipelineTestParameters.DataView);
			LogResult(nameof(AlgorithmType.LbfgsMultiClassifier), result);

			pipelineTransformer = pipelineParameters.Train(AlgorithmType.StochasticDualCoordinateAscentMultiClassifier);
			result = pipelineTransformer.EvaluateClustering(pipelineTestParameters.DataView);
			LogResult(nameof(AlgorithmType.StochasticDualCoordinateAscentMultiClassifier), result);
		}

		private void LogResult(string algorithm, ClusteringMetrics clusteringMetrics)
		{
			Console.WriteLine($"------------- {algorithm} - EVALUATION RESULTS -------------");
			Console.WriteLine($"AVG MIN SCORE = {clusteringMetrics.AverageDistance}");
			Console.WriteLine($"DBI = {clusteringMetrics.DaviesBouldinIndex}");
			Console.WriteLine($"NMI = {clusteringMetrics.NormalizedMutualInformation}");
			Console.WriteLine($"------------- {algorithm} - END EVALUATION -------------");
		}

		private ITrain GetPipelineParameters(string dataPath)
		{
			return (new Pipeline<CarData>(dataPath, _separator))
				.CopyColumn("Label", "Type")
				.ConcatenateColumns(_concatenatedColumns);
		}
	}
}
