using System.IO;
using System.Linq;
using System.Reflection;
using MachineLearningLibrary.Models;
using MachineLearningLibrary.Services;
using Microsoft.ML.Trainers;
using NUnit.Framework;

namespace MachineLearningLibraryTests
{
	[TestFixture]
	public class MulticlassClassificationTests
	{
		private PredictionService predictionService = new PredictionService();
		private string _dataPath;
		private char _separator;

		[SetUp]
		public void Setup()
		{
			var dir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
			_dataPath = $@"{dir}\data\taxi.csv";
			_separator = ',';
		}

		[Test]
		[TestCase(5.1f, 3.5f, 1.4f, 0.2f, "Iris-setosa")]
		[TestCase(4.9f, 3.0f, 1.4f, 0.2f, "Iris-setosa")]
		[TestCase(4.7f, 3.2f, 1.3f, 0.2f, "Iris-setosa")]
		[TestCase(7.0f, 3.2f, 4.7f, 1.4f, "Iris-versicolor")]
		[TestCase(6.4f, 3.2f, 4.5f, 1.5f, "Iris-versicolor")]
		[TestCase(6.9f, 3.1f, 4.9f, 1.5f, "Iris-versicolor")]
		[TestCase(6.3f, 3.3f, 6.0f, 2.5f, "Iris-virginica")]
		[TestCase(5.8f, 2.7f, 5.1f, 1.9f, "Iris-virginica")]
		[TestCase(7.1f, 3.0f, 5.9f, 2.1f, "Iris-virginica")]
		public void NaiveBayesClassifierTest(float sepalLength, float sepalWidth, float petalLenght, float petalWidth, string label)
		{
			var irisdata = new IrisData() { SepalLength = sepalLength, SepalWidth = sepalWidth, PetalLength = petalLenght, PetalWidth = petalWidth };
			var result = predictionService.MulticlassClassification<IrisData, IrisTypePrediction>(irisdata, new NaiveBayesClassifier(), _dataPath, _separator);
			Assert.AreEqual(result.Scores.OrderByDescending(s => s.Score).First().Label, label);
		}

		[Test]
		[TestCase(5.1f, 3.5f, 1.4f, 0.2f, "Iris-setosa")]
		[TestCase(4.9f, 3.0f, 1.4f, 0.2f, "Iris-setosa")]
		[TestCase(4.7f, 3.2f, 1.3f, 0.2f, "Iris-setosa")]
		[TestCase(7.0f, 3.2f, 4.7f, 1.4f, "Iris-versicolor")]
		[TestCase(6.4f, 3.2f, 4.5f, 1.5f, "Iris-versicolor")]
		[TestCase(6.9f, 3.1f, 4.9f, 1.5f, "Iris-versicolor")]
		[TestCase(6.3f, 3.3f, 6.0f, 2.5f, "Iris-virginica")]
		[TestCase(5.8f, 2.7f, 5.1f, 1.9f, "Iris-virginica")]
		[TestCase(7.1f, 3.0f, 5.9f, 2.1f, "Iris-virginica")]
		public void LogisticRegressionClassifierTest(float sepalLength, float sepalWidth, float petalLenght, float petalWidth, string label)
		{
			var irisdata = new IrisData() { SepalLength = sepalLength, SepalWidth = sepalWidth, PetalLength = petalLenght, PetalWidth = petalWidth };
			var result = predictionService.MulticlassClassification<IrisData, IrisTypePrediction>(irisdata, new LogisticRegressionClassifier(), _dataPath, _separator);
			Assert.AreEqual(result.Scores.OrderByDescending(s => s.Score).First().Label, label);
		}

		[Test]
		[TestCase(5.1f, 3.5f, 1.4f, 0.2f, "Iris-setosa")]
		[TestCase(4.9f, 3.0f, 1.4f, 0.2f, "Iris-setosa")]
		[TestCase(4.7f, 3.2f, 1.3f, 0.2f, "Iris-setosa")]
		[TestCase(7.0f, 3.2f, 4.7f, 1.4f, "Iris-versicolor")]
		[TestCase(6.4f, 3.2f, 4.5f, 1.5f, "Iris-versicolor")]
		[TestCase(6.9f, 3.1f, 4.9f, 1.5f, "Iris-versicolor")]
		[TestCase(6.3f, 3.3f, 6.0f, 2.5f, "Iris-virginica")]
		[TestCase(5.8f, 2.7f, 5.1f, 1.9f, "Iris-virginica")]
		[TestCase(7.1f, 3.0f, 5.9f, 2.1f, "Iris-virginica")]
		public void StochasticDualCoordinateAscentClassifierTest(float sepalLength, float sepalWidth, float petalLenght, float petalWidth, string label)
		{
			var irisdata = new IrisData() { SepalLength = sepalLength, SepalWidth = sepalWidth, PetalLength = petalLenght, PetalWidth = petalWidth };
			var result = predictionService.MulticlassClassification<IrisData, IrisTypePrediction>(irisdata, new StochasticDualCoordinateAscentClassifier(), _dataPath, _separator);
			Assert.AreEqual(result.Scores.OrderByDescending(s => s.Score).First().Label, label);
		}
	}
}
