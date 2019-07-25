﻿using System;
using System.IO;
using System.Reflection;
using MachineLearningLibrary.Models;
using MachineLearningLibrary.Services;
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
			var pipeline = new Pipeline2<GlassData>(_dataPath, _separator, AlgorithmType.NaiveBayesMultiClassifier, (_predictedColumn, true, null), _concatenatedColumns);
			var pipelineTest = new Pipeline2<GlassData>(_testDataPath, _separator);
			pipeline.BuildModel();

			var result = pipeline.EvaluateClustering(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.NaiveBayesMultiClassifier), result);

			pipeline = new Pipeline2<GlassData>(_dataPath, _separator, AlgorithmType.LbfgsMultiClassifier, (_predictedColumn, true, null), _concatenatedColumns);
			pipeline.BuildModel();
			result = pipeline.EvaluateClustering(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.LbfgsMultiClassifier), result);

			pipeline = new Pipeline2<GlassData>(_dataPath, _separator, AlgorithmType.StochasticDualCoordinateAscentMultiClassifier, (_predictedColumn, true, null), _concatenatedColumns);
			pipeline.BuildModel();
			result = pipeline.EvaluateClustering(pipelineTest.DataView);
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
	}
}
