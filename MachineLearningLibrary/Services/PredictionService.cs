using MachineLearningLibrary.Models;
using System;
using System.IO;
using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Linq;

namespace MachineLearningLibrary.Services
{
	public enum AlgorithmType
	{
		StochasticDualCoordinateAscentRegressor,
		FastTreeRegressor,
		FastTreeTweedieRegressor,
		FastForestRegressor,
		OnlineGradientDescentRegressor,
		PoissonRegressor,
		NaiveBayesMultiClassifier,
		StochasticDualCoordinateAscentMultiClassifier,
		FastForestBinaryClassifier,
		AveragedPerceptronBinaryClassifier,
		FastTreeBinaryClassifier,
		FieldAwareFactorizationMachineBinaryClassifier,
		LinearSvmBinaryClassifier,
		StochasticDualCoordinateAscentBinaryClassifier,
		StochasticGradientDescentBinaryClassifier,
		LbfgsMultiClassifier,
		GamBinaryClassifier,
		LbfgsBinaryClassifier
	}

	public class PredictionService
	{
		private readonly string _modelsRootPath;

		public PredictionService()
		{
			_modelsRootPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\Models";
			Directory.CreateDirectory(_modelsRootPath);
		}

		//public ITransformer Train<T, TPredictionModel>(Pipeline<T> pipelineParameters, AlgorithmType algorithmType) 
		//	where T : class
		//	where TPredictionModel : class, IPredictionModel, new()
		//{
		//	var modelPath = $@"{_modelsRootPath}\{Guid.NewGuid()}.zip";
		//	var model = GetModel(pipelineParameters, algorithmType);

		//	using (var fileStream = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
		//	{
		//		pipelineParameters.MlContext.Model.Save(model, pipelineParameters.DataView.Schema, fileStream);
		//	}

		//	return model;
		//}

		public RegressionMetrics EvaluateRegression<T>(ITransformer model, Pipeline<T> pipelineParameters, Pipeline<T> pipelineTestParameters) where T : class
		{
			return pipelineParameters.MlContext.Regression.Evaluate(model.Transform(pipelineTestParameters.DataView));
		}

		public BinaryClassificationMetrics EvaluateBinaryClassification<T>(ITransformer model, Pipeline<T> pipelineParameters, Pipeline<T> pipelineTestParameters) where T : class
		{
			return pipelineParameters.MlContext.BinaryClassification.Evaluate(model.Transform(pipelineTestParameters.DataView));
		}

		public ClusteringMetrics EvaluateClassification<T>(ITransformer model, Pipeline<T> pipelineParameters, Pipeline<T> pipelineTestParameters) where T : class
		{
			return pipelineParameters.MlContext.Clustering.Evaluate(model.Transform(pipelineTestParameters.DataView));
		}

		public TPredictionModel PredictScore<T, TPredictionModel>(T data, Pipeline<T> pipelineParameters, ITransformer model) 
			where T : class
			where TPredictionModel : class, IPredictionModel, new()
		{
			var predictionEngine = pipelineParameters.MlContext.Model.CreatePredictionEngine<T, TPredictionModel>(model);
			return predictionEngine.Predict(data);
		}

		//public async Task<ScoreLabel[]> PredictScoresAsync<T, TPrediction>(T data, string modelPath) where T : class where TPrediction : MultiClassificationPrediction, new()
		//{
		//	var model = await PredictionModel.ReadAsync<T, TPrediction>(modelPath);
		//	var prediction = model.Predict(data);
		//	model.TryGetScoreLabelNames(out string[] scoresLabels);

		//	return scoresLabels.Select(ls => new ScoreLabel()
		//	{
		//		Label = ls,
		//		Score = prediction.Scores[Array.IndexOf(scoresLabels, ls)]
		//	}).ToArray();
		//}

		//private ITransformer GetModel<T>(Pipeline<T> pipelineParameters, AlgorithmType algorithmType) where T : class
		//{
		//	switch (algorithmType)
		//	{
		//		case AlgorithmType.FastTreeRegressor:
		//			return pipelineParameters.Chain.Append(pipelineParameters.MlContext.Regression.Trainers.FastTree()).Fit(pipelineParameters.DataView);
		//		case AlgorithmType.StochasticDualCoordinateAscentRegressor:
		//			return pipelineParameters.Chain.Append(pipelineParameters.MlContext.Regression.Trainers.Sdca()).Fit(pipelineParameters.DataView);
		//		case AlgorithmType.FastTreeTweedieRegressor:
		//			return pipelineParameters.Chain.Append(pipelineParameters.MlContext.Regression.Trainers.FastTreeTweedie()).Fit(pipelineParameters.DataView);
		//		case AlgorithmType.FastForestRegressor:
		//			return pipelineParameters.Chain.Append(pipelineParameters.MlContext.Regression.Trainers.FastForest()).Fit(pipelineParameters.DataView);
		//		case AlgorithmType.OnlineGradientDescentRegressor:
		//			return pipelineParameters.Chain.Append(pipelineParameters.MlContext.Regression.Trainers.OnlineGradientDescent()).Fit(pipelineParameters.DataView);
		//		case AlgorithmType.PoissonRegressor:
		//			return pipelineParameters.Chain.Append(pipelineParameters.MlContext.Regression.Trainers.LbfgsPoissonRegression()).Fit(pipelineParameters.DataView);
		//		case AlgorithmType.NaiveBayesMultiClassifier:
		//			return pipelineParameters.Chain.Append(pipelineParameters.MlContext.MulticlassClassification.Trainers.NaiveBayes()).Fit(pipelineParameters.DataView);
		//		case AlgorithmType.StochasticDualCoordinateAscentMultiClassifier:
		//			return pipelineParameters.Chain.Append(pipelineParameters.MlContext.MulticlassClassification.Trainers.SdcaNonCalibrated()).Fit(pipelineParameters.DataView);
		//		case AlgorithmType.LbfgsMultiClassifier:
		//			return pipelineParameters.Chain.Append(pipelineParameters.MlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy()).Fit(pipelineParameters.DataView);
		//		case AlgorithmType.FastForestBinaryClassifier:
		//			return pipelineParameters.Chain.Append(pipelineParameters.MlContext.BinaryClassification.Trainers.FastForest(numberOfTrees: 3000)).Fit(pipelineParameters.DataView);
		//		case AlgorithmType.AveragedPerceptronBinaryClassifier:
		//			return pipelineParameters.Chain.Append(pipelineParameters.MlContext.BinaryClassification.Trainers.AveragedPerceptron()).Fit(pipelineParameters.DataView);
		//		case AlgorithmType.FastTreeBinaryClassifier:
		//			return pipelineParameters.Chain.Append(pipelineParameters.MlContext.BinaryClassification.Trainers.FastTree()).Fit(pipelineParameters.DataView);
		//		case AlgorithmType.FieldAwareFactorizationMachineBinaryClassifier:
		//			return pipelineParameters.Chain.Append(pipelineParameters.MlContext.BinaryClassification.Trainers.FieldAwareFactorizationMachine(pipelineParameters.ConcatenatedColumns.ToArray())).Fit(pipelineParameters.DataView);
		//		case AlgorithmType.LinearSvmBinaryClassifier:
		//			return pipelineParameters.Chain.Append(pipelineParameters.MlContext.BinaryClassification.Trainers.LinearSvm()).Fit(pipelineParameters.DataView);
		//		case AlgorithmType.StochasticDualCoordinateAscentBinaryClassifier:
		//			return pipelineParameters.Chain.Append(pipelineParameters.MlContext.BinaryClassification.Trainers.SdcaLogisticRegression()).Fit(pipelineParameters.DataView);
		//		case AlgorithmType.StochasticGradientDescentBinaryClassifier:
		//			return pipelineParameters.Chain.Append(pipelineParameters.MlContext.BinaryClassification.Trainers.SgdNonCalibrated()).Fit(pipelineParameters.DataView);
		//		case AlgorithmType.GamBinaryClassifier:
		//			return pipelineParameters.Chain.Append(pipelineParameters.MlContext.BinaryClassification.Trainers.Gam()).Fit(pipelineParameters.DataView);
		//		case AlgorithmType.LbfgsBinaryClassifier:
		//			return pipelineParameters.Chain.Append(pipelineParameters.MlContext.BinaryClassification.Trainers.LbfgsLogisticRegression()).Fit(pipelineParameters.DataView);
		//		default:
		//			return null;
		//	}
		//}
	}
}
