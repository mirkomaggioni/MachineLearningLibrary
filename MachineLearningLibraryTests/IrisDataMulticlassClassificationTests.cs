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
	public class IrisDataMulticlassClassificationTests
	{
		private PredictionService predictionService = new PredictionService();
		private string _dataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\traindata\iris.csv";
		private string _testDataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\testdata\iris.csv";
		private char _separator = ',';
		private string[] _concatenatedColumns = new[] { "SepalLength", "SepalWidth", "PetalLength", "PetalWidth" };

		[Test]
		public void IrisDataMulticlassClassificationTest()
		{
			var pipelineParameters = GetPipelineParameters(_dataPath);
			var pipelineTestParameters = GetPipelineParameters(_testDataPath);

			var modelPath = predictionService.Train<IrisData, IrisTypePrediction>(pipelineParameters, AlgorithmType.NaiveBayesClassifier);
			var result = predictionService.EvaluateClassification(pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.NaiveBayesClassifier), result);

			modelPath = predictionService.Train<IrisData, IrisTypePrediction>(pipelineParameters, AlgorithmType.LogisticRegressionClassifier);
			result = predictionService.EvaluateClassification(pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.LogisticRegressionClassifier), result);

			modelPath = predictionService.Train<IrisData, IrisTypePrediction>(pipelineParameters, AlgorithmType.StochasticDualCoordinateAscentClassifier);
			result = predictionService.EvaluateClassification(pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.StochasticDualCoordinateAscentClassifier), result);
		}

		private void LogResult(string algorithm, ClusteringMetrics clusteringMetrics)
		{
			Console.WriteLine($"------------- {algorithm} - EVALUATION RESULTS -------------");
			Console.WriteLine($"AVG MIN SCORE = {clusteringMetrics.AvgMinScore}");
			Console.WriteLine($"DBI = {clusteringMetrics.Dbi}");
			Console.WriteLine($"NMI = {clusteringMetrics.Nmi}");
			Console.WriteLine($"------------- {algorithm} - END EVALUATION -------------");
		}

		private PipelineParameters<IrisData> GetPipelineParameters(string dataPath)
		{
			return new PipelineParameters<IrisData>(dataPath, _separator, null, _concatenatedColumns);
		}
	}
}
