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
	public class GlassDataMulticlassClassificationTests
	{
		private readonly PredictionService predictionService = new PredictionService();
		private readonly string _dataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\traindata\glass.csv";
		private readonly string _testDataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\testdata\glass.csv";
		private readonly char _separator = ',';
		private readonly string _predictedColumn = "Type";
		private readonly string[] _concatenatedColumns = new[] { "IdNumber", "RefractiveIndex", "Sodium", "Magnesium", "Aluminium", "Silicon", "Potassium", "Calcium", "Barium", "Iron" };

		[Test]
		public void GlassDataMulticlassClassificationTest()
		{
			var pipelineParameters = GetPipelineParameters(_dataPath);
			var pipelineTestParameters = GetPipelineParameters(_testDataPath);

			var model = predictionService.Train<GlassData, GlassTypePrediction>(pipelineParameters, AlgorithmType.NaiveBayesMultiClassifier);
			var result = predictionService.EvaluateClassification(model, pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.NaiveBayesMultiClassifier), result);

			model = predictionService.Train<GlassData, GlassTypePrediction>(pipelineParameters, AlgorithmType.LbfgsMultiClassifier);
			result = predictionService.EvaluateClassification(model, pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.LbfgsMultiClassifier), result);

			model = predictionService.Train<GlassData, GlassTypePrediction>(pipelineParameters, AlgorithmType.StochasticDualCoordinateAscentMultiClassifier);
			result = predictionService.EvaluateClassification(model, pipelineParameters, pipelineTestParameters);
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

		private PipelineParameters<GlassData> GetPipelineParameters(string dataPath) {
			return new PipelineParameters<GlassData>(dataPath, _separator, (_predictedColumn, false), _concatenatedColumns, _concatenatedColumns);
		}
	}
}
