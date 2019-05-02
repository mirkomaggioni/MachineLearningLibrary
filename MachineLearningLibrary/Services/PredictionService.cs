using MachineLearningLibrary.Models;
using System;
using System.IO;
using System.Reflection;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

namespace MachineLearningLibrary.Services
{
	public class PredictionService
	{
		private readonly string _modelsRootPath;

		public PredictionService()
		{
			_modelsRootPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\Models";
			Directory.CreateDirectory(_modelsRootPath);
		}

		public PredictionEngine<T, TPredictionModel> Train<T, TTransformer, TModel, TPredictionModel>(ITrainerEstimator<TTransformer, TModel> trainerEstimator, PipelineParameters<T> pipelineParameters) 
			where T : class
			where TTransformer : ISingleFeaturePredictionTransformer<TModel>
			where TModel : class
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
			var model = trainerEstimator.Fit(pipelineParameters.DataView);

			using (var fileStream = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
			{
				pipelineParameters.MlContext.Model.Save(model, fileStream);
			}

			return pipelineParameters.MlContext.Model.CreatePredictionEngine<T, TPredictionModel>(model);
		}

		public RegressionMetrics EvaluateRegression<T>(PipelineParameters<T> pipelineParameters) where T : class
		{
			return pipelineParameters.MlContext.Regression.Evaluate(pipelineParameters.DataView);
		}

		public BinaryClassificationMetrics EvaluateBinaryClassification<T>(PipelineParameters<T> pipelineParameters) where T : class
		{
			return pipelineParameters.MlContext.BinaryClassification.Evaluate(pipelineParameters.DataView);
		}

		public ClusteringMetrics EvaluateClassification<T>(PipelineParameters<T> pipelineParameters) where T : class
		{
			return pipelineParameters.MlContext.Clustering.Evaluate(pipelineParameters.DataView);
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
	}
}
