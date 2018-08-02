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
		public void PriceEstimatorTest()
		{
			var car = new Car() { Manufacturer = "0", Color = "0", Year = "2017" };
			var result = predictionService.PricePrediction(car);
			Assert.IsNotNull(result);
			Assert.AreNotEqual(result.Price, 0);
			Assert.AreNotEqual(result.Price, 1);
		}
	}
}
