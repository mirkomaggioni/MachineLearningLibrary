﻿using MachineLearningLibrary.Models;
using System;
using System.IO;
using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;

namespace MachineLearningLibrary.Services
{
	public enum AlgorithmType
	{
		StochasticDualCoordinateAscent,
		FastTree,
		FastTreeTweedieRegressor,
		FastForestRegressor,
		OnlineGradientDescentRegressor,
		PoissonRegressor
	}

	public class PredictionService
	{
		private readonly string _modelsRootPath;

		public PredictionService()
		{
			_modelsRootPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\Models";
			Directory.CreateDirectory(_modelsRootPath);
		}

		public PredictionEngine<T, TPredictionModel> Train<T, TPredictionModel>(PipelineParameters<T> pipelineParameters, AlgorithmType algorithmType) 
			where T : class
			where TPredictionModel : class, IPredictionModel, new()
		{
			//if (pipelineParameters.MlContext..Algorithm == null)
			//	throw new ArgumentNullException(nameof(pipelineParameters.Algorithm));

			//var pipeline = new LearningPipeline();

			//if (pipelineParameters.TextLoader != null)
			//	pipeline.Add(pipelineParameters.TextLoader);

			//if (pipelineParameters.Dictionarizer != null)
			//	pipeline.Add(pipelineParameters.Dictionarizer);

			//if (pipelineParameters.CategoricalOneHotVectorizer != null)
			//	pipeline.Add(pipelineParameters.CategoricalOneHotVectorizer);

			//if (pipelineParameters.ColumnConcatenator != null)
			//	pipeline.Add(pipelineParameters.ColumnConcatenator);

			//pipeline.Add(pipelineParameters.Algorithm);

			//if (pipelineParameters.PredictedLabelColumnOriginalValueConverter != null)
			//	pipeline.Add(pipelineParameters.PredictedLabelColumnOriginalValueConverter);

			var modelPath = $@"{_modelsRootPath}\{Guid.NewGuid()}.zip";
			var model = GetModel(pipelineParameters, AlgorithmType.FastTree);

			using (var fileStream = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
			{
				pipelineParameters.MlContext.Model.Save(model, fileStream);
			}

			return pipelineParameters.MlContext.Model.CreatePredictionEngine<T, TPredictionModel>(model);
		}

		public RegressionMetrics EvaluateRegression<T>(PipelineParameters<T> pipelineParameters, PipelineParameters<T> pipelineTestParameters) where T : class
		{
			return pipelineParameters.MlContext.Regression.Evaluate(pipelineTestParameters.DataView);
		}

		public BinaryClassificationMetrics EvaluateBinaryClassification<T>(PipelineParameters<T> pipelineParameters, PipelineParameters<T> pipelineTestParameters) where T : class
		{
			return pipelineParameters.MlContext.BinaryClassification.Evaluate(pipelineTestParameters.DataView);
		}

		public ClusteringMetrics EvaluateClassification<T>(PipelineParameters<T> pipelineParameters, PipelineParameters<T> pipelineTestParameters) where T : class
		{
			return pipelineParameters.MlContext.Clustering.Evaluate(pipelineTestParameters.DataView);
		}

		public TPredictionModel PredictScore<T, TPredictionModel>(T data, PredictionEngine<T, TPredictionModel> predictionEngine) 
			where T : class
			where TPredictionModel : class, IPredictionModel, new()
		{
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
				case AlgorithmType.FastTree:
					return pipelineParameters.TextFeaturizingEstimator.Append(pipelineParameters.MlContext.Regression.Trainers.FastTree()).Fit(pipelineParameters.DataView);
				case AlgorithmType.StochasticDualCoordinateAscent:
					return pipelineParameters.TextFeaturizingEstimator.Append(pipelineParameters.MlContext.Regression.Trainers.StochasticDualCoordinateAscent()).Fit(pipelineParameters.DataView);
				case AlgorithmType.FastTreeTweedieRegressor:
					return pipelineParameters.TextFeaturizingEstimator.Append(pipelineParameters.MlContext.Regression.Trainers.FastTreeTweedie()).Fit(pipelineParameters.DataView);
				case AlgorithmType.FastForestRegressor:
					return pipelineParameters.TextFeaturizingEstimator.Append(pipelineParameters.MlContext.Regression.Trainers.FastForest()).Fit(pipelineParameters.DataView);
				case AlgorithmType.OnlineGradientDescentRegressor:
					return pipelineParameters.TextFeaturizingEstimator.Append(pipelineParameters.MlContext.Regression.Trainers.OnlineGradientDescent()).Fit(pipelineParameters.DataView);
				case AlgorithmType.PoissonRegressor:
					return pipelineParameters.TextFeaturizingEstimator.Append(pipelineParameters.MlContext.Regression.Trainers.PoissonRegression()).Fit(pipelineParameters.DataView);
				default:
					return null;
			}
		}
	}
}
