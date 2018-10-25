using MachineLearningLibrary.Models;
using Microsoft.ML.Legacy;
using Microsoft.ML.Legacy.Models;
using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;

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

		public async Task<string> TrainAsync<T, TPrediction>(PipelineParameters<T> pipelineParameters) where T : class where TPrediction : class, new()
		{
			if (pipelineParameters.Algorithm == null)
				throw new ArgumentNullException(nameof(pipelineParameters.Algorithm));

			var pipeline = new LearningPipeline();

			if (pipelineParameters.TextLoader != null)
				pipeline.Add(pipelineParameters.TextLoader);

			if (pipelineParameters.Dictionarizer != null)
				pipeline.Add(pipelineParameters.Dictionarizer);

			if (pipelineParameters.CategoricalOneHotVectorizer != null)
				pipeline.Add(pipelineParameters.CategoricalOneHotVectorizer);

			if (pipelineParameters.ColumnConcatenator != null)
				pipeline.Add(pipelineParameters.ColumnConcatenator);

			pipeline.Add(pipelineParameters.Algorithm);

			if (pipelineParameters.PredictedLabelColumnOriginalValueConverter != null)
				pipeline.Add(pipelineParameters.PredictedLabelColumnOriginalValueConverter);

			var modelPath = $@"{_modelsRootPath}\{Guid.NewGuid()}.zip";
			var model = pipeline.Train<T, TPrediction>();
			await model.WriteAsync(modelPath);
			return modelPath;
		}

		public async Task<RegressionMetrics> EvaluateRegressionAsync<T, TPrediction>(PipelineParameters<T> pipelineParameters, string modelPath) where T : class where TPrediction : class, new()
		{
			var model = await PredictionModel.ReadAsync<T, TPrediction>(modelPath);
			var regressionEvaluator = new RegressionEvaluator();
			return regressionEvaluator.Evaluate(model, pipelineParameters.TextLoader);
		}
	
		public async Task<BinaryClassificationMetrics> EvaluateBinaryClassificationAsync<T, TPrediction>(PipelineParameters<T> pipelineParameters, string modelPath) where T : class where TPrediction : class, new()
		{
			var model = await PredictionModel.ReadAsync<T, TPrediction>(modelPath);
			var binaryClassificationEvaluator = new BinaryClassificationEvaluator();
			return binaryClassificationEvaluator.Evaluate(model, pipelineParameters.TextLoader);
		}

		public async Task<ClassificationMetrics> EvaluateClassificationAsync<T, TPrediction>(PipelineParameters<T> pipelineParameters, string modelPath) where T : class where TPrediction : class, new()
		{
			var model = await PredictionModel.ReadAsync<T, TPrediction>(modelPath);
			var classificationEvaluator = new ClassificationEvaluator();
			return classificationEvaluator.Evaluate(model, pipelineParameters.TextLoader);
		}

		public async Task<TPrediction> PredictScoreAsync<T, TPrediction>(T data, string modelPath) where T : class where TPrediction : RegressionPrediction, new()
		{
			var model = await PredictionModel.ReadAsync<T, TPrediction>(modelPath);
			return model.Predict(data);
		}

		public async Task<ScoreLabel[]> PredictScoresAsync<T, TPrediction>(T data, string modelPath) where T : class where TPrediction : MultiClassificationPrediction, new()
		{
			var model = await PredictionModel.ReadAsync<T, TPrediction>(modelPath);
			var prediction = model.Predict(data);
			model.TryGetScoreLabelNames(out string[] scoresLabels);

			return scoresLabels.Select(ls => new ScoreLabel()
			{
				Label = ls,
				Score = prediction.Scores[Array.IndexOf(scoresLabels, ls)]
			}).ToArray();
		}
	}
}
