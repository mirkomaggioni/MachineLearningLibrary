using Microsoft.ML.Data;

namespace MachineLearningLibrary.Models
{
	public class CarData
	{
		[LoadColumn(0)]
		public float Symboling;

		[LoadColumn(1)]
		public float NormalizedLosses;

		[LoadColumn(2)]
		public string Make;

		[LoadColumn(3)]
		public string FuelType;

		[LoadColumn(4)]
		public string Aspiration;

		[LoadColumn(5)]
		public string Doors;

		[LoadColumn(6)]
		public string BodyStyle;

		[LoadColumn(7)]
		public string DriveWheels;

		[LoadColumn(8)]
		public string EngineLocation;

		[LoadColumn(9)]
		public float WheelBase;

		[LoadColumn(10)]
		public float Length;

		[LoadColumn(11)]
		public float Width;

		[LoadColumn(12)]
		public float Height;

		[LoadColumn(13)]
		public float CurbWeight;

		[LoadColumn(14)]
		public string EngineType;

		[LoadColumn(15)]
		public string NumOfCylinders;

		[LoadColumn(16)]
		public float EngineSize;

		[LoadColumn(17)]
		public string FuelSystem;

		[LoadColumn(18)]
		public float Bore;

		[LoadColumn(19)]
		public float Stroke;

		[LoadColumn(20)]
		public float CompressionRatio;

		[LoadColumn(21)]
		public float HorsePower;

		[LoadColumn(22)]
		public float PeakRpm;

		[LoadColumn(23)]
		public float CityMpg;

		[LoadColumn(24)]
		public float HighwayMpg;

		[LoadColumn(25), ColumnName("Label")]
		public float Price;
	}

	public class CarPricePrediction : RegressionPrediction { }
}
