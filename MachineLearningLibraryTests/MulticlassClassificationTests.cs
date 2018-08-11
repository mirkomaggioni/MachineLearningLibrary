using System.IO;
using System.Linq;
using System.Reflection;
using System.Threading.Tasks;
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
		private string _trainDataPath;
		private char _separator;
		private string _predictedColumn;
		private string[] _dictionarizedLabels;
		private string[] _concatenatedColumns;

		[SetUp]
		public void Setup()
		{
			var dir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
			_trainDataPath = $@"{dir}\traindata\iris.txt";
			_separator = ',';
			_predictedColumn = "PredictedLabel";
			_dictionarizedLabels = new[] { "Label" };
			_concatenatedColumns = new[] { "SepalLength", "SepalWidth", "PetalLength", "PetalWidth" };
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
		public async Task NaiveBayesClassifierTestAsync(float sepalLength, float sepalWidth, float petalLenght, float petalWidth, string label)
		{
			var irisdata = new IrisData() { SepalLength = sepalLength, SepalWidth = sepalWidth, PetalLength = petalLenght, PetalWidth = petalWidth };
			var modelPath = await predictionService.TrainAsync<IrisData, IrisTypePrediction, NaiveBayesClassifier>(_trainDataPath, _separator, _dictionarizedLabels, null, _concatenatedColumns, _predictedColumn);
			var result = await predictionService.PredictScoresAsync<IrisData, IrisTypePrediction>(irisdata, modelPath);

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
		public async Task LogisticRegressionClassifierTestAsync(float sepalLength, float sepalWidth, float petalLenght, float petalWidth, string label)
		{
			var irisdata = new IrisData() { SepalLength = sepalLength, SepalWidth = sepalWidth, PetalLength = petalLenght, PetalWidth = petalWidth };
			var modelPath = await predictionService.TrainAsync<IrisData, IrisTypePrediction, LogisticRegressionClassifier>(_trainDataPath, _separator, _dictionarizedLabels, null, _concatenatedColumns, _predictedColumn);
			var result = await predictionService.PredictScoresAsync<IrisData, IrisTypePrediction>(irisdata, modelPath);

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
		public async Task StochasticDualCoordinateAscentClassifierTestAsync(float sepalLength, float sepalWidth, float petalLenght, float petalWidth, string label)
		{
			var irisdata = new IrisData() { SepalLength = sepalLength, SepalWidth = sepalWidth, PetalLength = petalLenght, PetalWidth = petalWidth };
			var modelPath = await predictionService.TrainAsync<IrisData, IrisTypePrediction, StochasticDualCoordinateAscentClassifier>(_trainDataPath, _separator, _dictionarizedLabels, null, _concatenatedColumns, _predictedColumn);
			var result = await predictionService.PredictScoresAsync<IrisData, IrisTypePrediction>(irisdata, modelPath);

			Assert.AreEqual(result.Scores.OrderByDescending(s => s.Score).First().Label, label);
		}
	}
}
