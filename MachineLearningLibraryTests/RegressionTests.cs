using MachineLearningLibrary.Models;
using MachineLearningLibrary.Services;
using Microsoft.ML.Trainers;
using NUnit.Framework;

namespace MachineLearningLibraryTests
{
	[TestFixture]
	public class RegressionTests
	{
		private PredictionService predictionService = new PredictionService();

		[Test]
		[TestCase(0, 0, 2018, 38000)]
		[TestCase(0, 1, 2016, 28000)]
		[TestCase(2, 0, 2018, 28000)]
		[TestCase(3, 0, 2017, 13000)]
		public void StochasticDualCoordinateAscentRegressorTest(float manufacturer, float color, float year, float minValue)
		{
			var car = new CarData() { Manufacturer = manufacturer, Color = color, Year = year };
			var result = predictionService.Regression<CarData, CarPricePrediction>(car, new StochasticDualCoordinateAscentRegressor());
			Assert.IsTrue(result.Score > minValue);
		}

		[Test]
		[TestCase(0, 0, 2018, 38000)]
		[TestCase(0, 1, 2016, 28000)]
		[TestCase(2, 0, 2018, 28000)]
		[TestCase(3, 0, 2017, 13000)]
		public void FastTreeRegressorTest(float manufacturer, float color, float year, float minValue)
		{
			var car = new CarData() { Manufacturer = manufacturer, Color = color, Year = year };
			var result = predictionService.Regression<CarData, CarPricePrediction>(car,new FastTreeRegressor());
			Assert.IsTrue(result.Score > minValue);
		}

		[Test]
		[TestCase(0, 0, 2018, 38000)]
		[TestCase(0, 1, 2016, 28000)]
		[TestCase(2, 0, 2018, 28000)]
		[TestCase(3, 0, 2017, 13000)]
		public void FastTreeTweedieRegressorTest(float manufacturer, float color, float year, float minValue)
		{
			var car = new CarData() { Manufacturer = manufacturer, Color = color, Year = year };
			var result = predictionService.Regression<CarData, CarPricePrediction>(car, new FastTreeTweedieRegressor());
			Assert.IsTrue(result.Score > minValue);
		}

		[Test]
		[TestCase(0, 0, 2018, 38000)]
		[TestCase(0, 1, 2016, 28000)]
		[TestCase(2, 0, 2018, 28000)]
		[TestCase(3, 0, 2017, 13000)]
		public void FastForestRegressorTest(float manufacturer, float color, float year, float minValue)
		{
			var car = new CarData() { Manufacturer = manufacturer, Color = color, Year = year };
			var result = predictionService.Regression<CarData, CarPricePrediction>(car, new FastForestRegressor());
			Assert.IsTrue(result.Score > minValue);
		}

		[Test]
		[TestCase(0, 0, 2018, 38000)]
		[TestCase(0, 1, 2016, 28000)]
		[TestCase(2, 0, 2018, 28000)]
		[TestCase(3, 0, 2017, 13000)]
		public void OnlineGradientDescentRegressorTest(float manufacturer, float color, float year, float minValue)
		{
			var car = new CarData() { Manufacturer = manufacturer, Color = color, Year = year };
			var result = predictionService.Regression<CarData, CarPricePrediction>(car, new OnlineGradientDescentRegressor());
			Assert.IsTrue(result.Score > minValue);
		}

		[Test]
		[TestCase(0, 0, 2018, 38000)]
		[TestCase(0, 1, 2016, 28000)]
		[TestCase(2, 0, 2018, 28000)]
		[TestCase(3, 0, 2017, 13000)]
		public void PoissonRegressorTest(float manufacturer, float color, float year, float minValue)
		{
			var car = new CarData() { Manufacturer = manufacturer, Color = color, Year = year };
			var result = predictionService.Regression<CarData, CarPricePrediction>(car, new PoissonRegressor());
			Assert.IsTrue(result.Score > minValue);
		}

		[Test]
		[TestCase(0, 0, 2018, 38000)]
		[TestCase(0, 1, 2016, 28000)]
		[TestCase(2, 0, 2018, 28000)]
		[TestCase(3, 0, 2017, 13000)]
		public void GeneralizedAdditiveModelRegressorTest(float manufacturer, float color, float year, float minValue)
		{
			var car = new CarData() { Manufacturer = manufacturer, Color = color, Year = year };
			var result = predictionService.Regression<CarData, CarPricePrediction>(car, new GeneralizedAdditiveModelRegressor());
			Assert.IsTrue(result.Score > minValue);
		}
	}
}
