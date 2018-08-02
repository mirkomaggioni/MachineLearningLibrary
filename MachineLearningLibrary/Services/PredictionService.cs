using MachineLearningLibrary.Models;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System.IO;
using System.Reflection;

namespace MachineLearningLibrary.Services
{
	public enum MultiClassificationType
	{
		StochasticDualCoordinateAscentClassifier,
		LogisticRegressorClassifier,
		NaiveBayesClassifier
	}

	public class PredictionService<T,TPrediction> where T : class where TPrediction : class, new()
	{
		public CarPricePrediction PricePrediction(Car car)
		{
			var pipeline = new LearningPipeline();
			var dir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
			pipeline.Add(new TextLoader($@"{dir}\cars.txt").CreateFrom<Car>(separator: ';'));
			pipeline.Add(new ColumnCopier(("Price", "Label")));
			pipeline.Add(new CategoricalOneHotVectorizer("Manufacturer", "Color", "Year"));
			pipeline.Add(new ColumnConcatenator("Features", "Manufacturer", "Color", "Year"));
			pipeline.Add(new FastTreeRegressor());

			var model = pipeline.Train<Car, CarPricePrediction>();
			return model.Predict(car);
		}

		public TPrediction MulticlassClassification(T irisData, MultiClassificationType? type = MultiClassificationType.NaiveBayesClassifier)
		{
			var pipeline = new LearningPipeline();
			var dir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
			pipeline.Add(new TextLoader($@"{dir}\irisdata.txt").CreateFrom<T>(separator: ','));
			pipeline.Add(new Dictionarizer("Label"));
			pipeline.Add(new ColumnConcatenator("Features", "SepalLength", "SepalWidth", "PetalLength", "PetalWidth"));

			switch (type)
			{
				case MultiClassificationType.StochasticDualCoordinateAscentClassifier:
					pipeline.Add(new StochasticDualCoordinateAscentClassifier());
					break;

				case MultiClassificationType.LogisticRegressorClassifier:
					pipeline.Add(new LogisticRegressionClassifier());
					break;

				default:
					pipeline.Add(new NaiveBayesClassifier());
					break;
			}

			pipeline.Add(new PredictedLabelColumnOriginalValueConverter() { PredictedLabelColumn = "PredictedLabel" });

			var model = pipeline.Train<T, TPrediction>();
			return model.Predict(irisData);
		}
	}
}
