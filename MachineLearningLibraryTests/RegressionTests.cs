using MachineLearningLibrary.Models;
using MachineLearningLibrary.Services;
using NUnit.Framework;

namespace MachineLearningLibraryTests
{
	[TestFixture]
	public class RegressionTests
	{
		private PredictionService<IrisData, IrisLabelPrediction> predictionService = new PredictionService<IrisData, IrisLabelPrediction>();

		[Test]
		[TestCase("0", "0", "2018", 38000)]
		[TestCase("0", "1", "2016", 28000)]
		[TestCase("2", "0", "2018", 28000)]
		[TestCase("3", "0", "2017", 13000)]
		public void StochasticDualCoordinateAscentRegressorTest(string manufacturer, string color, string year, float minValue)
		{
			var car = new Car() { Manufacturer = manufacturer, Color = color, Year = year };
			var result = predictionService.Regression(car);
			Assert.IsTrue(result.Price > minValue);
		}
	}
}
