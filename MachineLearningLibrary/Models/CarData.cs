using Microsoft.ML.Runtime.Api;

namespace MachineLearningLibrary.Models
{
	public class CarData
	{
		[Column("0")]
		public float Symboling;

		[Column("1")]
		public float NormalizedLosses;

		[Column("2")]
		public string Make;

		[Column("3")]
		public string FuelType;

		[Column("4")]
		public string Aspiration;

		[Column("5")]
		public string Doors;

		[Column("6")]
		public string BodyStyle;

		[Column("7")]
		public string DriveWheels;

		[Column("8")]
		public string EngineLocation;

		[Column("9")]
		public float WheelBase;

		[Column("10")]
		public float Length;

		[Column("11")]
		public float Width;

		[Column("12")]
		public float Height;

		[Column("13")]
		public float CurbWeight;

		[Column("14")]
		public string EngineType;

		[Column("15")]
		public string NumOfCylinders;

		[Column("16")]
		public float EngineSize;

		[Column("17")]
		public string FuelSystem;

		[Column("18")]
		public float Bore;

		[Column("19")]
		public float Stroke;

		[Column("20")]
		public float CompressionRatio;

		[Column("21")]
		public float HorsePower;

		[Column("22")]
		public float PeakRpm;

		[Column("23")]
		public float CityMpg;

		[Column("24")]
		public float HighwayMpg;

		[Column("25", "Label")]
		public float Price;
	}

	public class CarPricePrediction : SingleScore { }
}
