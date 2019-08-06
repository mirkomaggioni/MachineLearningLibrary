using System;
using System.IO;
using System.Reflection;
using MachineLearningLibrary.Models;
using NUnit.Framework;
using Microsoft.ML.Data;

namespace MachineLearningLibraryTests
{
	[TestFixture]
	public class CarDataRegressionEvaluationTests
	{
		private readonly string _dataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\traindata\car.csv";
		private readonly string _testDataPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\testdata\car.csv";
		private readonly char _separator = ',';
		private readonly string _predictedColumn = "Price";
		private readonly string[] _alphanumericColumns = new[] { "Make", "FuelType", "Aspiration", "Doors", "BodyStyle", "DriveWheels", "EngineLocation", "EngineType", "NumOfCylinders", "FuelSystem" };
		private readonly string[] _concatenatedColumns = new[] { "Symboling", "NormalizedLosses", "Make", "FuelType", "Aspiration", "Doors", "BodyStyle", "DriveWheels", "EngineLocation", "WheelBase", "Length", "Width", "Height", "CurbWeight", "EngineType", "NumOfCylinders", "EngineSize", "FuelSystem",
												"Bore", "Stroke", "CompressionRatio", "HorsePower", "PeakRpm", "CityMpg", "HighwayMpg"};

		[Test]
		public void CarDataRegressionEvaluationTest()
		{
			var pipeline = new Pipeline<CarData>(_dataPath, _separator, AlgorithmType.StochasticDualCoordinateAscentRegressor, (_predictedColumn, false, null), _concatenatedColumns, _alphanumericColumns);
			var pipelineTest = new Pipeline<CarData>(_testDataPath, _separator);
			pipeline.BuildModel();

			var result = pipeline.EvaluateRegression(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.StochasticDualCoordinateAscentRegressor), result);

			pipeline = new Pipeline<CarData>(_dataPath, _separator, AlgorithmType.FastTreeRegressor, (_predictedColumn, false, null), _concatenatedColumns, _alphanumericColumns);
			pipeline.BuildModel();
			result = pipeline.EvaluateRegression(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.FastTreeRegressor), result);

			pipeline = new Pipeline<CarData>(_dataPath, _separator, AlgorithmType.FastTreeTweedieRegressor, (_predictedColumn, false, null), _concatenatedColumns, _alphanumericColumns);
			pipeline.BuildModel();
			result = pipeline.EvaluateRegression(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.FastTreeTweedieRegressor), result);

			pipeline = new Pipeline<CarData>(_dataPath, _separator, AlgorithmType.FastForestRegressor, (_predictedColumn, false, null), _concatenatedColumns, _alphanumericColumns);
			pipeline.BuildModel();
			result = pipeline.EvaluateRegression(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.FastForestRegressor), result);

			//pipeline = new Pipeline2<CarData>(_dataPath, _separator, AlgorithmType.OnlineGradientDescentRegressor, (_predictedColumn, false), _concatenatedColumns, _alphanumericColumns);
			//pipeline.BuildModel();
			//result = pipeline.EvaluateRegression(pipelineTest.DataView);
			//LogResult(nameof(AlgorithmType.OnlineGradientDescentRegressor), result);

			pipeline = new Pipeline<CarData>(_dataPath, _separator, AlgorithmType.PoissonRegressor, (_predictedColumn, false, null), _concatenatedColumns, _alphanumericColumns);
			pipeline.BuildModel();
			result = pipeline.EvaluateRegression(pipelineTest.DataView);
			LogResult(nameof(AlgorithmType.PoissonRegressor), result);
		}

		[Test]
		[TestCase(0f, 102f, "subaru", "gas", "std", "four", "sedan", "fwd", "front", 97.20f, 172.00f, 65.40f, 52.50f, 2145f, "ohcf", "four", 108f, "2bbl", 3.62f, 2.64f, 9.50f, 82f, 4800f, 32f, 37f, 7126f)]
		[TestCase(0f, 161f, "peugot", "gas", "std", "four", "sedan", "rwd", "front", 107.90f, 186.70f, 68.40f, 56.70f, 3075f, "l", "four", 120f, "mpfi", 3.46f, 2.19f, 8.40f, 95f, 5000f, 19f, 24f, 15580f)]
		public void GlassDataClusteringPredictTest(float symboling, float normalizedLosses, string make, string fuelType, string aspiration, string doors, string bodyStyle, string driveWheels, string engineLocation, float wheelBase, float length, float width, float height, float curbWeight, string engineType, string numOfCylinders, float engineSize, string fuelSystem, float bore, float stroke, float compressionRatio, float horsePower, float peakRpm, float cityMpg, float highwayMpg, float price)
		{
			var pipeline = new Pipeline<CarData>(_dataPath, _separator, AlgorithmType.StochasticDualCoordinateAscentRegressor, (_predictedColumn, false, null), _concatenatedColumns, _alphanumericColumns);
			pipeline.BuildModel();

			var prediction = pipeline.PredictScore<CarData, CarPricePrediction, float>(new CarData() { Symboling = symboling, NormalizedLosses = normalizedLosses, Make = make, FuelType = fuelType, Aspiration = aspiration, Doors = doors, BodyStyle = bodyStyle, DriveWheels = driveWheels, EngineLocation = engineLocation, WheelBase = wheelBase, Length = length, Width = width, Height = height, CurbWeight = curbWeight, EngineType = engineType, NumOfCylinders = numOfCylinders, EngineSize = engineSize, FuelSystem = fuelSystem, Bore = bore, Stroke = stroke, CompressionRatio = compressionRatio, HorsePower = horsePower, PeakRpm = peakRpm, CityMpg = cityMpg, HighwayMpg = highwayMpg });

			Assert.AreEqual(prediction.Score, price);
		}

		private void LogResult(string algorithm, RegressionMetrics regressionMetrics)
		{
			Console.WriteLine($"------------- {algorithm} - EVALUATION RESULTS -------------");
			Console.WriteLine($"RMS = {regressionMetrics.RootMeanSquaredError}");
			Console.WriteLine($"RSQUARED = {regressionMetrics.RSquared}");
			Console.WriteLine($"------------- {algorithm} - END EVALUATION -------------");
		}
	}
}
