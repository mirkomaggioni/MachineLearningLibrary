using MachineLearningLibrary.Models;
using Microsoft.ML;
using Microsoft.ML.Data;
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

		public async Task<string> TrainAsync<T, TPrediction, TAlgorythm>(string dataPath, char separator, string[] dictionarizedLabels, string[] alphanumericColums, string[] concatenatedColumns, string predictedColumn = null) where T : class where TPrediction : class, new() where TAlgorythm : ILearningPipelineItem, new()
		{
			var pipeline = new LearningPipeline();
			pipeline.Add(new TextLoader(dataPath).CreateFrom<T>(separator: separator));
			
			if (dictionarizedLabels != null)
				pipeline.Add(new Dictionarizer(dictionarizedLabels));

			if (alphanumericColums != null)
				pipeline.Add(new CategoricalOneHotVectorizer(alphanumericColums));

			pipeline.Add(new ColumnConcatenator("Features", concatenatedColumns));
			pipeline.Add(new TAlgorythm());

			if (!string.IsNullOrEmpty(predictedColumn))
				pipeline.Add(new PredictedLabelColumnOriginalValueConverter { PredictedLabelColumn = predictedColumn });

			var modelPath = $@"{_modelsRootPath}\{Guid.NewGuid()}.zip";
			var model = pipeline.Train<T, TPrediction>();
			await model.WriteAsync(modelPath);
			return modelPath;
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
