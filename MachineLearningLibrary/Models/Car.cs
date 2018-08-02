using Microsoft.ML.Runtime.Api;

namespace MachineLearningLibrary.Models
{
	public enum Manufacturer
	{
		Bmw,
		Mercedes,
		Volkswagen,
		Fiat,
		Kia
	}

	public enum Color
	{
		White,
		Grey,
		Black,
		Red
	}

	public class Car
	{
		[Column("0")]
		[VectorType(1000)]
		public uint[] Manufacturer;
		[Column("1")]
		[VectorType(1000)]
		public uint[] Color;
		[Column("2")]
		[VectorType(1000)]
		public uint[] Year;
		[Column("3", "Label")]
		public float Price;
	}

	public class CarPricePrediction
	{
		[ColumnName("PredictedPrice")]
		public float PredictedPrices;
	}
}
