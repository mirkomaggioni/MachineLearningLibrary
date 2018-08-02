using MachineLearningLibrary.Models;
using MachineLearningLibrary.Services;
using NUnit.Framework;
using System;

namespace MachineLearningLibraryTests
{
	[TestFixture]
	public class PredictionServiceTests
	{
		private PredictionService predictionService = new PredictionService();

		[Test]
		public void PriceEstimatorTest()
		{
			var car = new Car() { Color = new uint[] {0}, Manufacturer = new uint[] { 0 }, Year = new uint[] { 2017 } };
			var result = predictionService.PricePrediction(car);
			Assert.IsNotNull(result);
		}
	}
}
