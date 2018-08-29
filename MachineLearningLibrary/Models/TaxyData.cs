using Microsoft.ML.Runtime.Api;

namespace MachineLearningLibrary.Models
{
	public class TaxyData
	{
		[Column("0")]
		public string VendorId;

		[Column("1")]
		public string RateCode;

		[Column("2")]
		public float PassengerCount;

		[Column("3")]
		public float TripTime;

		[Column("4")]
		public float TripDistance;

		[Column("5")]
		public string PaymentType;

		[Column("6")]
		public float FareAmount;
	}

	public class TaxyTripFarePrediction : SingleScore {}
}
