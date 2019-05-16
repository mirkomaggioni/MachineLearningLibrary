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
		private readonly PredictionService predictionService = new PredictionService();
		private readonly string _dataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\traindata\iris.csv";
		private readonly string _testDataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\testdata\iris.csv";
		private readonly char _separator = ',';
		private readonly string _predictedColumn = "Type";
		private readonly string[] _concatenatedColumns = new[] { "SepalLength", "SepalWidth", "PetalLength", "PetalWidth" };

		[Test]
		public void IrisDataMulticlassClassificationTest()
		{
			var pipelineParameters = GetPipelineParameters(_dataPath);
			var pipelineTestParameters = GetPipelineParameters(_testDataPath);

			var model = predictionService.Train<IrisData, IrisTypePrediction>(pipelineParameters, AlgorithmType.NaiveBayesMultiClassifier);
			var result = predictionService.EvaluateClassification(model, pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.NaiveBayesMultiClassifier), result);

			model = predictionService.Train<IrisData, IrisTypePrediction>(pipelineParameters, AlgorithmType.LogisticRegressionMultiClassifier);
			result = predictionService.EvaluateClassification(model, pipelineParameters, pipelineTestParameters);
			LogResult(nameof(AlgorithmType.LogisticRegressionMultiClassifier), result);

			model = predictionService.Train<IrisData, IrisTypePrediction>(pipelineParameters, AlgorithmType.StochasticDualCoordinateAscentMultiClassifier);
			result = predictionService.EvaluateClassification(model, pipelineParameters, pipelineTestParameters);
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

		private PipelineParameters<IrisData> GetPipelineParameters(string dataPath)
		{
			return new PipelineParameters<IrisData>(dataPath, _separator, _predictedColumn, null, _concatenatedColumns);
		}
	}
}
