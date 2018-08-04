using MachineLearningLibrary.Models;
using MachineLearningLibrary.Services;
using Microsoft.ML.Trainers;
using NUnit.Framework;

namespace MachineLearningLibraryTests
{
	[TestFixture]
	public class RegressionTests
	{
		private PredictionService<IrisData, IrisTypePrediction> predictionService = new PredictionService<IrisData, IrisTypePrediction>();

		[Test]
		[TestCase("0", "0", "2018", 38000)]
		[TestCase("0", "1", "2016", 28000)]
		[TestCase("2", "0", "2018", 28000)]
		[TestCase("3", "0", "2017", 13000)]
		public void StochasticDualCoordinateAscentRegressorTest(string manufacturer, string color, string year, float minValue)
		{
			var car = new Car() { Manufacturer = manufacturer, Color = color, Year = year };
			var result = predictionService.Regression(car, new StochasticDualCoordinateAscentRegressor());
			Assert.IsTrue(result.Price > minValue);
		}

		[Test]
		[TestCase("0", "0", "2018", 38000)]
		[TestCase("0", "1", "2016", 28000)]
		[TestCase("2", "0", "2018", 28000)]
		[TestCase("3", "0", "2017", 13000)]
		public void FastTreeRegressorTest(string manufacturer, string color, string year, float minValue)
		{
			var car = new Car() { Manufacturer = manufacturer, Color = color, Year = year };
			var result = predictionService.Regression(car,new FastTreeRegressor());
			Assert.IsTrue(result.Price > minValue);
		}

		[Test]
		[TestCase("0", "0", "2018", 38000)]
		[TestCase("0", "1", "2016", 28000)]
		[TestCase("2", "0", "2018", 28000)]
		[TestCase("3", "0", "2017", 13000)]
		public void FastTreeTweedieRegressorTest(string manufacturer, string color, string year, float minValue)
		{
			var car = new Car() { Manufacturer = manufacturer, Color = color, Year = year };
			var result = predictionService.Regression(car, new FastTreeTweedieRegressor());
			Assert.IsTrue(result.Price > minValue);
		}

		[Test]
		[TestCase("0", "0", "2018", 38000)]
		[TestCase("0", "1", "2016", 28000)]
		[TestCase("2", "0", "2018", 28000)]
		[TestCase("3", "0", "2017", 13000)]
		public void FastForestRegressorTest(string manufacturer, string color, string year, float minValue)
		{
			var car = new Car() { Manufacturer = manufacturer, Color = color, Year = year };
			var result = predictionService.Regression(car, new FastForestRegressor());
			Assert.IsTrue(result.Price > minValue);
		}

		[Test]
		[TestCase("0", "0", "2018", 38000)]
		[TestCase("0", "1", "2016", 28000)]
		[TestCase("2", "0", "2018", 28000)]
		[TestCase("3", "0", "2017", 13000)]
		public void OnlineGradientDescentRegressorTest(string manufacturer, string color, string year, float minValue)
		{
			var car = new Car() { Manufacturer = manufacturer, Color = color, Year = year };
			var result = predictionService.Regression(car, new OnlineGradientDescentRegressor());
			Assert.IsTrue(result.Price > minValue);
		}

		[Test]
		[TestCase("0", "0", "2018", 38000)]
		[TestCase("0", "1", "2016", 28000)]
		[TestCase("2", "0", "2018", 28000)]
		[TestCase("3", "0", "2017", 13000)]
		public void PoissonRegressorTest(string manufacturer, string color, string year, float minValue)
		{
			var car = new Car() { Manufacturer = manufacturer, Color = color, Year = year };
			var result = predictionService.Regression(car, new PoissonRegressor());
			Assert.IsTrue(result.Price > minValue);
		}

		[Test]
		[TestCase("0", "0", "2018", 38000)]
		[TestCase("0", "1", "2016", 28000)]
		[TestCase("2", "0", "2018", 28000)]
		[TestCase("3", "0", "2017", 13000)]
		public void GeneralizedAdditiveModelRegressorTest(string manufacturer, string color, string year, float minValue)
		{
			var car = new Car() { Manufacturer = manufacturer, Color = color, Year = year };
			var result = predictionService.Regression(car, new GeneralizedAdditiveModelRegressor());
			Assert.IsTrue(result.Price > minValue);
		}
	}
}
