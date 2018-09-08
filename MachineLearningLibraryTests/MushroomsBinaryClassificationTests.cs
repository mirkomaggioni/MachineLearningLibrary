using System;
using System.IO;
using System.Reflection;
using System.Threading.Tasks;
using MachineLearningLibrary.Models;
using MachineLearningLibrary.Services;
using Microsoft.ML;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using NUnit.Framework;

namespace MachineLearningLibraryTests
{
	[TestFixture]
	public class MushroomsBinaryClassificationTests
	{
		private PredictionService predictionService = new PredictionService();
		private string _dataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\traindata\mushroom.csv";
		private string _testDataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\testdata\mushroom.csv";
		private char _separator = ',';
		private string _predictedLabel = "PredictedLabel";
		private string[] _dictionarizedLabels = new[] { "Label" };
		private string[] _concatenatedColumns = new[] { "CapShape", "CapSurface", "CapColor", "Bruises", "Odor", "GillAttachment", "GillSpacing", "GillSize", "GillColor", "StalkShape", "StalkRoot", "StalkSurfaceAboveRing", "StalkSurfaceBelowRing", "StalkColorAboveRing", "StalkColorBelowRing", "VeilType", "VeilColor", "RingNumber", "RingType", "SporePrintColor", "Population", "Habitat" };

		[Test]
		public async Task MushroomsBinaryClassificationTest()
		{
			var algorithm = new FastForestBinaryClassifier();
			algorithm.NumTrees = 3000;
			var pipelineParameters = GetPipelineParameters(algorithm);
			var pipelineTestParameters = GetPipelineTestParameters();
			var modelPath = await predictionService.TrainAsync<MushroomData, MushroomEdiblePrediction>(pipelineParameters);
			var result = await predictionService.EvaluateBinaryClassificationAsync<MushroomData, MushroomEdiblePrediction>(pipelineTestParameters, modelPath);
			LogResult(nameof(FastForestBinaryClassifier), result);

			pipelineParameters = GetPipelineParameters(new AveragedPerceptronBinaryClassifier());
			modelPath = await predictionService.TrainAsync<MushroomData, MushroomEdiblePrediction>(pipelineParameters);
			result = await predictionService.EvaluateBinaryClassificationAsync<MushroomData, MushroomEdiblePrediction>(pipelineTestParameters, modelPath);
			LogResult(nameof(AveragedPerceptronBinaryClassifier), result);

			pipelineParameters = GetPipelineParameters(new FastTreeBinaryClassifier());
			modelPath = await predictionService.TrainAsync<MushroomData, MushroomEdiblePrediction>(pipelineParameters);
			result = await predictionService.EvaluateBinaryClassificationAsync<MushroomData, MushroomEdiblePrediction>(pipelineTestParameters, modelPath);
			LogResult(nameof(FastTreeBinaryClassifier), result);

			pipelineParameters = GetPipelineParameters(new FieldAwareFactorizationMachineBinaryClassifier());
			modelPath = await predictionService.TrainAsync<MushroomData, MushroomEdiblePrediction>(pipelineParameters);
			result = await predictionService.EvaluateBinaryClassificationAsync<MushroomData, MushroomEdiblePrediction>(pipelineTestParameters, modelPath);
			LogResult(nameof(FieldAwareFactorizationMachineBinaryClassifier), result);

			pipelineParameters = GetPipelineParameters(new GeneralizedAdditiveModelBinaryClassifier());
			modelPath = await predictionService.TrainAsync<MushroomData, MushroomEdiblePrediction>(pipelineParameters);
			result = await predictionService.EvaluateBinaryClassificationAsync<MushroomData, MushroomEdiblePrediction>(pipelineTestParameters, modelPath);
			LogResult(nameof(GeneralizedAdditiveModelBinaryClassifier), result);

			pipelineParameters = GetPipelineParameters(new LinearSvmBinaryClassifier());
			modelPath = await predictionService.TrainAsync<MushroomData, MushroomEdiblePrediction>(pipelineParameters);
			result = await predictionService.EvaluateBinaryClassificationAsync<MushroomData, MushroomEdiblePrediction>(pipelineTestParameters, modelPath);
			LogResult(nameof(LinearSvmBinaryClassifier), result);

			pipelineParameters = GetPipelineParameters(new LogisticRegressionBinaryClassifier());
			modelPath = await predictionService.TrainAsync<MushroomData, MushroomEdiblePrediction>(pipelineParameters);
			result = await predictionService.EvaluateBinaryClassificationAsync<MushroomData, MushroomEdiblePrediction>(pipelineTestParameters, modelPath);
			LogResult(nameof(LogisticRegressionBinaryClassifier), result);

			pipelineParameters = GetPipelineParameters(new StochasticDualCoordinateAscentBinaryClassifier());
			modelPath = await predictionService.TrainAsync<MushroomData, MushroomEdiblePrediction>(pipelineParameters);
			result = await predictionService.EvaluateBinaryClassificationAsync<MushroomData, MushroomEdiblePrediction>(pipelineTestParameters, modelPath);
			LogResult(nameof(StochasticDualCoordinateAscentBinaryClassifier), result);

			pipelineParameters = GetPipelineParameters(new StochasticGradientDescentBinaryClassifier());
			modelPath = await predictionService.TrainAsync<MushroomData, MushroomEdiblePrediction>(pipelineParameters);
			result = await predictionService.EvaluateBinaryClassificationAsync<MushroomData, MushroomEdiblePrediction>(pipelineTestParameters, modelPath);
			LogResult(nameof(StochasticGradientDescentBinaryClassifier), result);
		}

		private void LogResult(string algorithm, BinaryClassificationMetrics classificationMetrics)
		{
			Console.WriteLine($"------------- {algorithm} - EVALUATION RESULTS -------------");
			Console.WriteLine($"Accuracy = {classificationMetrics.Accuracy}");
			Console.WriteLine($"Auc = {classificationMetrics.Auc}");
			Console.WriteLine($"F1Score = {classificationMetrics.F1Score}");
			Console.WriteLine($"LogLoss = {classificationMetrics.LogLoss}");
			Console.WriteLine($"LogLossReduction = {classificationMetrics.LogLossReduction}");
			Console.WriteLine($"------------- {algorithm} - END EVALUATION -------------");
		}

		private PipelineParameters<MushroomData> GetPipelineParameters(ILearningPipelineItem algorithm) {
			return new PipelineParameters<MushroomData>(_dataPath, _separator, _predictedLabel, _concatenatedColumns, _dictionarizedLabels, _concatenatedColumns, algorithm);
		}

		private PipelineParameters<MushroomData> GetPipelineTestParameters() {
			return new PipelineParameters<MushroomData>(_testDataPath, _separator);
		}
	}
}
