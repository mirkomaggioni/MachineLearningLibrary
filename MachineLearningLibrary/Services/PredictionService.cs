using MachineLearningLibrary.Models;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.IO;
using System.Linq;
using System.Reflection;

namespace MachineLearningLibrary.Services
{
	public class PredictionService
	{
		public TPrediction Regression<T, TPrediction>(T car, ILearningPipelineItem algorythm, string dataPath, char separator) where T : class where TPrediction : class, new()
		{
			var pipeline = new LearningPipeline();
			
			pipeline.Add(new TextLoader(dataPath).CreateFrom<T>(separator: separator));
			pipeline.Add(new CategoricalOneHotVectorizer("VendorId", "RateCode", "PaymentType"));
			pipeline.Add(new ColumnConcatenator("Features", "VendorId", "RateCode", "PassengerCount", "TripDistance", "PaymentType"));
			pipeline.Add(algorythm);

			var model = pipeline.Train<T, TPrediction>();
			return model.Predict(car);
		}

		public TPrediction MulticlassClassification<T, TPrediction>(T irisData, ILearningPipelineItem algorythm, string dataPath, char separator) where T : class where TPrediction : MultipleScores, new()
		{
			var pipeline = new LearningPipeline();
			var dir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
			pipeline.Add(new TextLoader($@"{dir}\data\iris.txt").CreateFrom<T>(separator: ','));
			pipeline.Add(new Dictionarizer("Label"));
			pipeline.Add(new ColumnConcatenator("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));
			pipeline.Add(algorythm);
			pipeline.Add(new PredictedLabelColumnOriginalValueConverter { PredictedLabelColumn = "PredictedLabel" });

			var model = pipeline.Train<T, TPrediction>();
			var prediction = model.Predict(irisData);
			model.TryGetScoreLabelNames(out string[] labelsScores);

			prediction.Scores = labelsScores.Select(ls => new ScoreLabel() {
				Label = ls,
				Score = prediction.Score[Array.IndexOf(labelsScores, ls)]
			}).ToArray();

			return prediction;
		}
	}
}
