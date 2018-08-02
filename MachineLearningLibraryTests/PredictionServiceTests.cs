using MachineLearningLibrary.Models;
using MachineLearningLibrary.Services;
using NUnit.Framework;

namespace MachineLearningLibraryTests
{
	[TestFixture]
	public class PredictionServiceTests
	{
		private PredictionService<IrisData, IrisLabelPrediction> predictionService = new PredictionService<IrisData, IrisLabelPrediction>();

		[Test]
		public void PriceEstimatorTest()
		{
			var car = new Car() { Manufacturer = "0", Color = "0", Year = "2017" };
			var result = predictionService.PricePrediction(car);
			Assert.IsNotNull(result);
			Assert.AreNotEqual(result.Price, 0);
			Assert.AreNotEqual(result.Price, 1);
		}

		[Test]
		[TestCase(3.3f, 1.6f, 0.2f, 5.1f, "Iris-setosa")]
		[TestCase(5.3f, 3.4f, 0.2f, 0.2f, "Iris-setosa")]
		[TestCase(4.2f, 1.4f, 1.5f, 0.7f, "Iris-setosa")]
		[TestCase(5.4f, 3.1f, 4.2f, 1.9f, "Iris-versicolor")]
		[TestCase(6.3f, 3.1f, 6.0f, 2.0f, "Iris-versicolor")]
		[TestCase(6.3f, 3.1f, 6.0f, 2.0f, "Iris-versicolor")]
		public void NaiveBayesClassifierTest(float sepalLength, float sepalWidth, float petalLenght, float petalWidth, string label)
		{
			var irisdata = new IrisData() { SepalLength = sepalLength, SepalWidth = sepalWidth, PetalLength = petalLenght, PetalWidth = petalWidth };
			var result = predictionService.MulticlassClassification(irisdata, MultiClassificationType.NaiveBayesClassifier);
			Assert.AreEqual(result.PredictedLabels, label);
		}
	}
}
