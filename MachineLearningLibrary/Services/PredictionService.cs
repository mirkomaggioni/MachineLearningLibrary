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
			var model = GetModel(pipelineParameters, AlgorithmType.FastTreeRegressor);

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
				case AlgorithmType.FastTreeRegressor:
					return pipelineParameters.TextFeaturizingEstimator.Append(pipelineParameters.MlContext.Regression.Trainers.FastTree()).Fit(pipelineParameters.DataView);
				case AlgorithmType.StochasticDualCoordinateAscentRegressor:
					return pipelineParameters.TextFeaturizingEstimator.Append(pipelineParameters.MlContext.Regression.Trainers.StochasticDualCoordinateAscent()).Fit(pipelineParameters.DataView);
				case AlgorithmType.FastTreeTweedieRegressor:
					return pipelineParameters.TextFeaturizingEstimator.Append(pipelineParameters.MlContext.Regression.Trainers.FastTreeTweedie()).Fit(pipelineParameters.DataView);
				case AlgorithmType.FastForestRegressor:
					return pipelineParameters.TextFeaturizingEstimator.Append(pipelineParameters.MlContext.Regression.Trainers.FastForest()).Fit(pipelineParameters.DataView);
				case AlgorithmType.OnlineGradientDescentRegressor:
					return pipelineParameters.TextFeaturizingEstimator.Append(pipelineParameters.MlContext.Regression.Trainers.OnlineGradientDescent()).Fit(pipelineParameters.DataView);
				case AlgorithmType.PoissonRegressor:
					return pipelineParameters.TextFeaturizingEstimator.Append(pipelineParameters.MlContext.Regression.Trainers.PoissonRegression()).Fit(pipelineParameters.DataView);
				case AlgorithmType.NaiveBayesMultiClassifier:
					return pipelineParameters.TextFeaturizingEstimator.Append(pipelineParameters.MlContext.MulticlassClassification.Trainers.NaiveBayes()).Fit(pipelineParameters.DataView);
				case AlgorithmType.LogisticRegressionMultiClassifier:
					return pipelineParameters.TextFeaturizingEstimator.Append(pipelineParameters.MlContext.MulticlassClassification.Trainers.LogisticRegression()).Fit(pipelineParameters.DataView);
				case AlgorithmType.StochasticDualCoordinateAscentMultiClassifier:
					return pipelineParameters.TextFeaturizingEstimator.Append(pipelineParameters.MlContext.MulticlassClassification.Trainers.StochasticDualCoordinateAscent()).Fit(pipelineParameters.DataView);
				case AlgorithmType.FastForestBinaryClassifier:
					return pipelineParameters.TextFeaturizingEstimator.Append(pipelineParameters.MlContext.BinaryClassification.Trainers.FastForest(numTrees: 3000)).Fit(pipelineParameters.DataView);
				case AlgorithmType.AveragedPerceptronBinaryClassifier:
					return pipelineParameters.TextFeaturizingEstimator.Append(pipelineParameters.MlContext.BinaryClassification.Trainers.AveragedPerceptron()).Fit(pipelineParameters.DataView);
				case AlgorithmType.FastTreeBinaryClassifier:
					return pipelineParameters.TextFeaturizingEstimator.Append(pipelineParameters.MlContext.BinaryClassification.Trainers.FastTree()).Fit(pipelineParameters.DataView);
				case AlgorithmType.FieldAwareFactorizationMachineBinaryClassifier:
					return pipelineParameters.TextFeaturizingEstimator.Append(pipelineParameters.MlContext.BinaryClassification.Trainers.FieldAwareFactorizationMachine(pipelineParameters.ConcatenatedColumns.ToArray())).Fit(pipelineParameters.DataView);
				case AlgorithmType.GeneralizedAdditiveModelBinaryClassifier:
					return pipelineParameters.TextFeaturizingEstimator.Append(pipelineParameters.MlContext.BinaryClassification.Trainers.GeneralizedAdditiveModels()).Fit(pipelineParameters.DataView);
				case AlgorithmType.LinearSvmBinaryClassifier:
					return pipelineParameters.TextFeaturizingEstimator.Append(pipelineParameters.MlContext.BinaryClassification.Trainers.LinearSupportVectorMachines()).Fit(pipelineParameters.DataView);
				case AlgorithmType.LogisticRegressionBinaryClassifier:
					return pipelineParameters.TextFeaturizingEstimator.Append(pipelineParameters.MlContext.BinaryClassification.Trainers.LogisticRegression()).Fit(pipelineParameters.DataView);
				case AlgorithmType.StochasticDualCoordinateAscentBinaryClassifier:
					return pipelineParameters.TextFeaturizingEstimator.Append(pipelineParameters.MlContext.BinaryClassification.Trainers.StochasticDualCoordinateAscent()).Fit(pipelineParameters.DataView);
				case AlgorithmType.StochasticGradientDescentBinaryClassifier:
					return pipelineParameters.TextFeaturizingEstimator.Append(pipelineParameters.MlContext.BinaryClassification.Trainers.StochasticGradientDescent()).Fit(pipelineParameters.DataView);
				default:
					return null;
			}
		}
	}
}
