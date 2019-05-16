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
		LogisticRegressionMultiClassifier,
		StochasticDualCoordinateAscentMultiClassifier,
		FastForestBinaryClassifier,
		AveragedPerceptronBinaryClassifier,
		FastTreeBinaryClassifier,
		FieldAwareFactorizationMachineBinaryClassifier,
		GeneralizedAdditiveModelBinaryClassifier,
		LinearSvmBinaryClassifier,
		LogisticRegressionBinaryClassifier,
		StochasticDualCoordinateAscentBinaryClassifier,
		StochasticGradientDescentBinaryClassifier
	}

	public class PredictionService
	{
		private readonly string _modelsRootPath;

		public PredictionService()
		{
			_modelsRootPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\Models";
			Directory.CreateDirectory(_modelsRootPath);
		}

		public ITransformer Train<T, TPredictionModel>(PipelineParameters<T> pipelineParameters, AlgorithmType algorithmType) 
			where T : class
			where TPredictionModel : class, IPredictionModel, new()
		{
			var modelPath = $@"{_modelsRootPath}\{Guid.NewGuid()}.zip";
			var model = GetModel(pipelineParameters, algorithmType);

			using (var fileStream = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
			{
				pipelineParameters.MlContext.Model.Save(model, fileStream);
			}

			return model;
		}

		public RegressionMetrics EvaluateRegression<T>(ITransformer model, PipelineParameters<T> pipelineParameters, PipelineParameters<T> pipelineTestParameters) where T : class
		{
			return pipelineParameters.MlContext.Regression.Evaluate(model.Transform(pipelineTestParameters.DataView));
		}

		public BinaryClassificationMetrics EvaluateBinaryClassification<T>(ITransformer model, PipelineParameters<T> pipelineParameters, PipelineParameters<T> pipelineTestParameters) where T : class
		{
			return pipelineParameters.MlContext.BinaryClassification.Evaluate(model.Transform(pipelineTestParameters.DataView));
		}

		public ClusteringMetrics EvaluateClassification<T>(ITransformer model, PipelineParameters<T> pipelineParameters, PipelineParameters<T> pipelineTestParameters) where T : class
		{
			return pipelineParameters.MlContext.Clustering.Evaluate(model.Transform(pipelineTestParameters.DataView));
		}

		public TPredictionModel PredictScore<T, TPredictionModel>(T data, PipelineParameters<T> pipelineParameters, ITransformer model) 
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

		private ITransformer GetModel<T>(PipelineParameters<T> pipelineParameters, AlgorithmType algorithmType) where T : class
		{
			switch (algorithmType)
			{
				case AlgorithmType.FastTreeRegressor:
					return pipelineParameters.ColumnCopyingEstimator.Append(pipelineParameters.MlContext.Regression.Trainers.FastTree()).Fit(pipelineParameters.DataView);
				case AlgorithmType.StochasticDualCoordinateAscentRegressor:
					return pipelineParameters.ColumnCopyingEstimator.Append(pipelineParameters.MlContext.Regression.Trainers.StochasticDualCoordinateAscent()).Fit(pipelineParameters.DataView);
				case AlgorithmType.FastTreeTweedieRegressor:
					return pipelineParameters.ColumnCopyingEstimator.Append(pipelineParameters.MlContext.Regression.Trainers.FastTreeTweedie()).Fit(pipelineParameters.DataView);
				case AlgorithmType.FastForestRegressor:
					return pipelineParameters.ColumnCopyingEstimator.Append(pipelineParameters.MlContext.Regression.Trainers.FastForest()).Fit(pipelineParameters.DataView);
				case AlgorithmType.OnlineGradientDescentRegressor:
					return pipelineParameters.ColumnCopyingEstimator.Append(pipelineParameters.MlContext.Regression.Trainers.OnlineGradientDescent()).Fit(pipelineParameters.DataView);
				case AlgorithmType.PoissonRegressor:
					return pipelineParameters.ColumnCopyingEstimator.Append(pipelineParameters.MlContext.Regression.Trainers.PoissonRegression()).Fit(pipelineParameters.DataView);
				case AlgorithmType.NaiveBayesMultiClassifier:
					return pipelineParameters.ColumnCopyingEstimator.Append(pipelineParameters.MlContext.MulticlassClassification.Trainers.NaiveBayes()).Fit(pipelineParameters.DataView);
				case AlgorithmType.LogisticRegressionMultiClassifier:
					return pipelineParameters.ColumnCopyingEstimator.Append(pipelineParameters.MlContext.MulticlassClassification.Trainers.LogisticRegression()).Fit(pipelineParameters.DataView);
				case AlgorithmType.StochasticDualCoordinateAscentMultiClassifier:
					return pipelineParameters.ColumnCopyingEstimator.Append(pipelineParameters.MlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent()).Fit(pipelineParameters.DataView);
				case AlgorithmType.FastForestBinaryClassifier:
					return pipelineParameters.ColumnCopyingEstimator.Append(pipelineParameters.MlContext.BinaryClassification.Trainers.FastForest(numTrees: 3000)).Fit(pipelineParameters.DataView);
				case AlgorithmType.AveragedPerceptronBinaryClassifier:
					return pipelineParameters.ColumnCopyingEstimator.Append(pipelineParameters.MlContext.BinaryClassification.Trainers.AveragedPerceptron()).Fit(pipelineParameters.DataView);
				case AlgorithmType.FastTreeBinaryClassifier:
					return pipelineParameters.ColumnCopyingEstimator.Append(pipelineParameters.MlContext.BinaryClassification.Trainers.FastTree()).Fit(pipelineParameters.DataView);
				case AlgorithmType.FieldAwareFactorizationMachineBinaryClassifier:
					return pipelineParameters.ColumnCopyingEstimator.Append(pipelineParameters.MlContext.BinaryClassification.Trainers.FieldAwareFactorizationMachine(pipelineParameters.ConcatenatedColumns.ToArray())).Fit(pipelineParameters.DataView);
				case AlgorithmType.GeneralizedAdditiveModelBinaryClassifier:
					return pipelineParameters.ColumnCopyingEstimator.Append(pipelineParameters.MlContext.BinaryClassification.Trainers.GeneralizedAdditiveModels()).Fit(pipelineParameters.DataView);
				case AlgorithmType.LinearSvmBinaryClassifier:
					return pipelineParameters.ColumnCopyingEstimator.Append(pipelineParameters.MlContext.BinaryClassification.Trainers.LinearSupportVectorMachines()).Fit(pipelineParameters.DataView);
				case AlgorithmType.LogisticRegressionBinaryClassifier:
					return pipelineParameters.ColumnCopyingEstimator.Append(pipelineParameters.MlContext.BinaryClassification.Trainers.LogisticRegression()).Fit(pipelineParameters.DataView);
				case AlgorithmType.StochasticDualCoordinateAscentBinaryClassifier:
					return pipelineParameters.ColumnCopyingEstimator.Append(pipelineParameters.MlContext.BinaryClassification.Trainers.StochasticDualCoordinateAscent()).Fit(pipelineParameters.DataView);
				case AlgorithmType.StochasticGradientDescentBinaryClassifier:
					return pipelineParameters.ColumnCopyingEstimator.Append(pipelineParameters.MlContext.BinaryClassification.Trainers.StochasticGradientDescent()).Fit(pipelineParameters.DataView);
				default:
					return null;
			}
		}
	}
}
