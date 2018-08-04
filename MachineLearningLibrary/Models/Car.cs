using Microsoft.ML.Runtime.Api;

namespace MachineLearningLibrary.Models
{
	public class Car
	{
		[Column("0")]
		public float Manufacturer;
		[Column("1")]
		public float Color;
		[Column("2")]
		public float Year;
		[Column("3", "Label")]
		public float Price;
	}

	public class CarPricePrediction
	{
		[ColumnName("Score")]
		public float Price;
	}
}
