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
		private PredictionService predictionService = new PredictionService();
		private string _dataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\traindata\glass.csv";
		private string _testDataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\testdata\glass.csv";
		private char _separator = ',';
		private string[] _concatenatedColumns = new[] { "IdNumber", "RefractiveIndex", "Sodium", "Magnesium", "Aluminium", "Silicon", "Potassium", "Calcium", "Barium", "Iron" };

		[Test]
		public void GlassDataMulticlassClassificationTest()
		{
			var pipelineParameters = GetPipelineParameters(_dataPath);
			var pipelineTestParameters = GetPipelineParameters(_testDataPath);

			var modelPath = predictionService.Train<GlassData, GlassTypePrediction>(pipelineParameters, AlgorithmType.NaiveBayesMultiClassifier);
			var result = predictionService.EvaluateClassification(pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.NaiveBayesMultiClassifier), result);

			modelPath = predictionService.Train<GlassData, GlassTypePrediction>(pipelineParameters, AlgorithmType.LogisticRegressionMultiClassifier);
			result = predictionService.EvaluateClassification(pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.LogisticRegressionMultiClassifier), result);

			modelPath = predictionService.Train<GlassData, GlassTypePrediction>(pipelineParameters, AlgorithmType.StochasticDualCoordinateAscentMultiClassifier);
			result = predictionService.EvaluateClassification(pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.StochasticDualCoordinateAscentMultiClassifier), result);
		}

		private void LogResult(string algorithm, ClusteringMetrics clusteringMetrics)
		{
			Console.WriteLine($"------------- {algorithm} - EVALUATION RESULTS -------------");
			Console.WriteLine($"AVG MIN SCORE = {clusteringMetrics.AvgMinScore}");
			Console.WriteLine($"DBI = {clusteringMetrics.Dbi}");
			Console.WriteLine($"NMI = {clusteringMetrics.Nmi}");
			Console.WriteLine($"------------- {algorithm} - END EVALUATION -------------");
		}

		private PipelineParameters<GlassData> GetPipelineParameters(string dataPath) {
			return new PipelineParameters<GlassData>(dataPath, _separator, null, _concatenatedColumns);
		}
	}
}
