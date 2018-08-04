using Microsoft.ML.Runtime.Api;

namespace MachineLearningLibrary.Models
{
	public class Car
	{
		[Column("0")]
		public string Manufacturer;
		[Column("1")]
		public string Color;
		[Column("2")]
		public string Year;
		[Column("3")]
		public float Price;
	}

	public class CarPricePrediction
	{
		[ColumnName("Score")]
		public float Price;
	}
}
