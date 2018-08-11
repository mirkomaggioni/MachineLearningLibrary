using System.IO;
using System.Reflection;
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
		[TestCase("VTS", "1", 1, 1140, 3.75f, "CRD", 15.5f)]
		[TestCase("VTS", "1", 1, 480, 2.72f, "CRD", 10.0f)]
		[TestCase("VTS", "1", 1, 1680, 7.8f, "CSH", 26.5f)]
		[TestCase("VTS", "1", 1, 600, 4.73f, "CSH", 14.5f)]
		public void StochasticDualCoordinateAscentRegressorTest(string vendorId, string rateCode, float passengerCount, float tripTime, float tripDistance, string paymentType, float fareAmount)
		{
			var car = new TaxyData() { VendorId = vendorId, RateCode = rateCode, PassengerCount = passengerCount, TripTime = tripTime, TripDistance = tripDistance, PaymentType = paymentType };
			var result = predictionService.Regression<TaxyData, TaxyTripFarePrediction>(car, new StochasticDualCoordinateAscentRegressor(), _dataPath, _separator);
			Assert.IsTrue(result.Score >= fareAmount);
		}

		[Test]
		[TestCase("VTS", "1", 1, 1140, 3.75f, "CRD", 15.5f)]
		[TestCase("VTS", "1", 1, 480, 2.72f, "CRD", 10.0f)]
		[TestCase("VTS", "1", 1, 1680, 7.8f, "CSH", 26.5f)]
		[TestCase("VTS", "1", 1, 600, 4.73f, "CSH", 14.5f)]
		public void FastTreeRegressorTest(string vendorId, string rateCode, float passengerCount, float tripTime, float tripDistance, string paymentType, float fareAmount)
		{
			var car = new TaxyData() { VendorId = vendorId, RateCode = rateCode, PassengerCount = passengerCount, TripTime = tripTime, TripDistance = tripDistance, PaymentType = paymentType };
			var result = predictionService.Regression<TaxyData, TaxyTripFarePrediction>(car,new FastTreeRegressor(), _dataPath, _separator);
			Assert.IsTrue(result.Score >= fareAmount);
		}

		[Test]
		[TestCase("VTS", "1", 1, 1140, 3.75f, "CRD", 15.5f)]
		[TestCase("VTS", "1", 1, 480, 2.72f, "CRD", 10.0f)]
		[TestCase("VTS", "1", 1, 1680, 7.8f, "CSH", 26.5f)]
		[TestCase("VTS", "1", 1, 600, 4.73f, "CSH", 14.5f)]
		public void FastTreeTweedieRegressorTest(string vendorId, string rateCode, float passengerCount, float tripTime, float tripDistance, string paymentType, float fareAmount)
		{
			var car = new TaxyData() { VendorId = vendorId, RateCode = rateCode, PassengerCount = passengerCount, TripTime = tripTime, TripDistance = tripDistance, PaymentType = paymentType };
			var result = predictionService.Regression<TaxyData, TaxyTripFarePrediction>(car, new FastTreeTweedieRegressor(), _dataPath, _separator);
			Assert.IsTrue(result.Score >= fareAmount);
		}

		[Test]
		[TestCase("VTS", "1", 1, 1140, 3.75f, "CRD", 15.5f)]
		[TestCase("VTS", "1", 1, 480, 2.72f, "CRD", 10.0f)]
		[TestCase("VTS", "1", 1, 1680, 7.8f, "CSH", 26.5f)]
		[TestCase("VTS", "1", 1, 600, 4.73f, "CSH", 14.5f)]
		public void FastForestRegressorTest(string vendorId, string rateCode, float passengerCount, float tripTime, float tripDistance, string paymentType, float fareAmount)
		{
			var car = new TaxyData() { VendorId = vendorId, RateCode = rateCode, PassengerCount = passengerCount, TripTime = tripTime, TripDistance = tripDistance, PaymentType = paymentType };
			var result = predictionService.Regression<TaxyData, TaxyTripFarePrediction>(car, new FastForestRegressor(), _dataPath, _separator);
			Assert.IsTrue(result.Score >= fareAmount);
		}

		[Test]
		[TestCase("VTS", "1", 1, 1140, 3.75f, "CRD", 15.5f)]
		[TestCase("VTS", "1", 1, 480, 2.72f, "CRD", 10.0f)]
		[TestCase("VTS", "1", 1, 1680, 7.8f, "CSH", 26.5f)]
		[TestCase("VTS", "1", 1, 600, 4.73f, "CSH", 14.5f)]
		public void OnlineGradientDescentRegressorTest(string vendorId, string rateCode, float passengerCount, float tripTime, float tripDistance, string paymentType, float fareAmount)
		{
			var car = new TaxyData() { VendorId = vendorId, RateCode = rateCode, PassengerCount = passengerCount, TripTime = tripTime, TripDistance = tripDistance, PaymentType = paymentType };
			var result = predictionService.Regression<TaxyData, TaxyTripFarePrediction>(car, new OnlineGradientDescentRegressor(), _dataPath, _separator);
			Assert.IsTrue(result.Score >= fareAmount);
		}

		[Test]
		[TestCase("VTS", "1", 1, 1140, 3.75f, "CRD", 15.5f)]
		[TestCase("VTS", "1", 1, 480, 2.72f, "CRD", 10.0f)]
		[TestCase("VTS", "1", 1, 1680, 7.8f, "CSH", 26.5f)]
		[TestCase("VTS", "1", 1, 600, 4.73f, "CSH", 14.5f)]
		public void PoissonRegressorTest(string vendorId, string rateCode, float passengerCount, float tripTime, float tripDistance, string paymentType, float fareAmount)
		{
			var car = new TaxyData() { VendorId = vendorId, RateCode = rateCode, PassengerCount = passengerCount, TripTime = tripTime, TripDistance = tripDistance, PaymentType = paymentType };
			var result = predictionService.Regression<TaxyData, TaxyTripFarePrediction>(car, new PoissonRegressor(), _dataPath, _separator);
			Assert.IsTrue(result.Score >= fareAmount);
		}
	}
}
