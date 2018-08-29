﻿using MachineLearningLibrary.Models;
using Microsoft.ML;
using Microsoft.ML.Models;
using Microsoft.ML.Transforms;
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

		public async Task<string> TrainAsync<T, TPrediction, TAlgorythm>(PipelineParameters<T> pipelineParameters) where T : class where TPrediction : class, new() where TAlgorythm : ILearningPipelineItem, new()
		{
			var pipeline = new LearningPipeline();

			if (pipelineParameters.TextLoader != null)
				pipeline.Add(pipelineParameters.TextLoader);

			if (pipelineParameters.Dictionarizer != null)
				pipeline.Add(pipelineParameters.Dictionarizer);

			if (pipelineParameters.CategoricalOneHotVectorizer != null)
				pipeline.Add(pipelineParameters.CategoricalOneHotVectorizer);

			if (pipelineParameters.ColumnConcatenator != null)
				pipeline.Add(pipelineParameters.ColumnConcatenator);

			pipeline.Add(new ColumnCopier((pipelineParameters.LabelColumn, "Label")));

			pipeline.Add(new TAlgorythm());

			if (pipelineParameters.PredictedLabelColumnOriginalValueConverter != null)
				pipeline.Add(pipelineParameters.PredictedLabelColumnOriginalValueConverter);

			var modelPath = $@"{_modelsRootPath}\{Guid.NewGuid()}.zip";
			var model = pipeline.Train<T, TPrediction>();
			await model.WriteAsync(modelPath);
			return modelPath;
		}

		public async Task<RegressionMetrics> EvaluateAsync<T, TPrediction>(PipelineParameters<T> pipelineParameters, string modelPath) where T : class where TPrediction : class, new()
		{
			var model = await PredictionModel.ReadAsync<T, TPrediction>(modelPath);
			var regressionEvaluator = new RegressionEvaluator();
			return regressionEvaluator.Evaluate(model, pipelineParameters.TextLoader);
		}

		public async Task<TPrediction> PredictScoreAsync<T, TPrediction>(T data, string modelPath) where T : class where TPrediction : SingleScore, new()
		{
			var model = await PredictionModel.ReadAsync<T, TPrediction>(modelPath);
			return model.Predict(data);
		}

		public async Task<TPrediction> PredictScoresAsync<T, TPrediction>(T data, string modelPath) where T : class where TPrediction : MultipleScores, new()
		{
			var model = await PredictionModel.ReadAsync<T, TPrediction>(modelPath);
			var prediction = model.Predict(data);
			model.TryGetScoreLabelNames(out string[] labelsScores);

			prediction.Scores = labelsScores.Select(ls => new ScoreLabel()
			{
				Label = ls,
				Score = prediction.Score[Array.IndexOf(labelsScores, ls)]
			}).ToArray();

			return prediction;
		}
	}
}
