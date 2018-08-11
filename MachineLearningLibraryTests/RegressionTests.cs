using System.IO;
using System.Reflection;
using System.Threading.Tasks;
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
		private string[] _alphanumericColumns;
		private string[] _concatenatedColumns;

		[SetUp]
		public void Setup()
		{
			var dir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
			_dataPath = $@"{dir}\traindata\taxi.csv";
			_separator = ',';
			_alphanumericColumns = new[] { "VendorId", "RateCode", "PaymentType" };
			_concatenatedColumns = new[] { "VendorId", "RateCode", "PassengerCount", "TripDistance", "PaymentType" };
		}

		[Test]
		[TestCase("VTS", "1", 1, 1140, 3.75f, "CRD", 15.5f)]
		[TestCase("VTS", "1", 1, 480, 2.72f, "CRD", 10.0f)]
		[TestCase("VTS", "1", 1, 1680, 7.8f, "CSH", 26.5f)]
		[TestCase("VTS", "1", 1, 600, 4.73f, "CSH", 14.5f)]
		public async Task StochasticDualCoordinateAscentRegressorTestAsync(string vendorId, string rateCode, float passengerCount, float tripTime, float tripDistance, string paymentType, float fareAmount)
		{
			var car = new TaxyData() { VendorId = vendorId, RateCode = rateCode, PassengerCount = passengerCount, TripTime = tripTime, TripDistance = tripDistance, PaymentType = paymentType };
			var modelPath = await predictionService.TrainAsync<TaxyData, TaxyTripFarePrediction, StochasticDualCoordinateAscentRegressor>(_dataPath, _separator, null, _alphanumericColumns, _concatenatedColumns, null);
			var result = await predictionService.PredictScoreAsync<TaxyData, TaxyTripFarePrediction>(car, modelPath);
			Assert.IsTrue(result.Score >= fareAmount);
		}

		[Test]
		[TestCase("VTS", "1", 1, 1140, 3.75f, "CRD", 15.5f)]
		[TestCase("VTS", "1", 1, 480, 2.72f, "CRD", 10.0f)]
		[TestCase("VTS", "1", 1, 1680, 7.8f, "CSH", 26.5f)]
		[TestCase("VTS", "1", 1, 600, 4.73f, "CSH", 14.5f)]
		public async Task FastTreeRegressorTestAsync(string vendorId, string rateCode, float passengerCount, float tripTime, float tripDistance, string paymentType, float fareAmount)
		{
			var car = new TaxyData() { VendorId = vendorId, RateCode = rateCode, PassengerCount = passengerCount, TripTime = tripTime, TripDistance = tripDistance, PaymentType = paymentType };
			var modelPath = await predictionService.TrainAsync<TaxyData, TaxyTripFarePrediction, StochasticDualCoordinateAscentRegressor>(_dataPath, _separator, null, _alphanumericColumns, _concatenatedColumns, null);
			var result = await predictionService.PredictScoreAsync<TaxyData, TaxyTripFarePrediction>(car, modelPath);
			Assert.IsTrue(result.Score >= fareAmount);
		}

		[Test]
		[TestCase("VTS", "1", 1, 1140, 3.75f, "CRD", 15.5f)]
		[TestCase("VTS", "1", 1, 480, 2.72f, "CRD", 10.0f)]
		[TestCase("VTS", "1", 1, 1680, 7.8f, "CSH", 26.5f)]
		[TestCase("VTS", "1", 1, 600, 4.73f, "CSH", 14.5f)]
		public async Task FastTreeTweedieRegressorTestAsync(string vendorId, string rateCode, float passengerCount, float tripTime, float tripDistance, string paymentType, float fareAmount)
		{
			var car = new TaxyData() { VendorId = vendorId, RateCode = rateCode, PassengerCount = passengerCount, TripTime = tripTime, TripDistance = tripDistance, PaymentType = paymentType };
			var modelPath = await predictionService.TrainAsync<TaxyData, TaxyTripFarePrediction, StochasticDualCoordinateAscentRegressor>(_dataPath, _separator, null, _alphanumericColumns, _concatenatedColumns, null);
			var result = await predictionService.PredictScoreAsync<TaxyData, TaxyTripFarePrediction>(car, modelPath);
			Assert.IsTrue(result.Score >= fareAmount);
		}

		[Test]
		[TestCase("VTS", "1", 1, 1140, 3.75f, "CRD", 15.5f)]
		[TestCase("VTS", "1", 1, 480, 2.72f, "CRD", 10.0f)]
		[TestCase("VTS", "1", 1, 1680, 7.8f, "CSH", 26.5f)]
		[TestCase("VTS", "1", 1, 600, 4.73f, "CSH", 14.5f)]
		public async Task FastForestRegressorTestAsync(string vendorId, string rateCode, float passengerCount, float tripTime, float tripDistance, string paymentType, float fareAmount)
		{
			var car = new TaxyData() { VendorId = vendorId, RateCode = rateCode, PassengerCount = passengerCount, TripTime = tripTime, TripDistance = tripDistance, PaymentType = paymentType };
			var modelPath = await predictionService.TrainAsync<TaxyData, TaxyTripFarePrediction, StochasticDualCoordinateAscentRegressor>(_dataPath, _separator, null, _alphanumericColumns, _concatenatedColumns, null);
			var result = await predictionService.PredictScoreAsync<TaxyData, TaxyTripFarePrediction>(car, modelPath);
			Assert.IsTrue(result.Score >= fareAmount);
		}

		[Test]
		[TestCase("VTS", "1", 1, 1140, 3.75f, "CRD", 15.5f)]
		[TestCase("VTS", "1", 1, 480, 2.72f, "CRD", 10.0f)]
		[TestCase("VTS", "1", 1, 1680, 7.8f, "CSH", 26.5f)]
		[TestCase("VTS", "1", 1, 600, 4.73f, "CSH", 14.5f)]
		public async Task OnlineGradientDescentRegressorTestAsync(string vendorId, string rateCode, float passengerCount, float tripTime, float tripDistance, string paymentType, float fareAmount)
		{
			var car = new TaxyData() { VendorId = vendorId, RateCode = rateCode, PassengerCount = passengerCount, TripTime = tripTime, TripDistance = tripDistance, PaymentType = paymentType };
			var modelPath = await predictionService.TrainAsync<TaxyData, TaxyTripFarePrediction, StochasticDualCoordinateAscentRegressor>(_dataPath, _separator, null, _alphanumericColumns, _concatenatedColumns, null);
			var result = await predictionService.PredictScoreAsync<TaxyData, TaxyTripFarePrediction>(car, modelPath);
			Assert.IsTrue(result.Score >= fareAmount);
		}

		[Test]
		[TestCase("VTS", "1", 1, 1140, 3.75f, "CRD", 15.5f)]
		[TestCase("VTS", "1", 1, 480, 2.72f, "CRD", 10.0f)]
		[TestCase("VTS", "1", 1, 1680, 7.8f, "CSH", 26.5f)]
		[TestCase("VTS", "1", 1, 600, 4.73f, "CSH", 14.5f)]
		public async Task PoissonRegressorTestAsync(string vendorId, string rateCode, float passengerCount, float tripTime, float tripDistance, string paymentType, float fareAmount)
		{
			var car = new TaxyData() { VendorId = vendorId, RateCode = rateCode, PassengerCount = passengerCount, TripTime = tripTime, TripDistance = tripDistance, PaymentType = paymentType };
			var modelPath = await predictionService.TrainAsync<TaxyData, TaxyTripFarePrediction, StochasticDualCoordinateAscentRegressor>(_dataPath, _separator, null, _alphanumericColumns, _concatenatedColumns, null);
			var result = await predictionService.PredictScoreAsync<TaxyData, TaxyTripFarePrediction>(car, modelPath);
			Assert.IsTrue(result.Score >= fareAmount);
		}
	}
}
