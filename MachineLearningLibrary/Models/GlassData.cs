using Microsoft.ML.Runtime.Api;

namespace MachineLearningLibrary.Models
{
	public class GlassData
	{
		[Column("0")]
		public float IdNumber;
		[Column("1")]
		public float RefractiveIndex;
		[Column("2")]
		public float Sodium;
		[Column("3")]
		public float Magnesium;
		[Column("4")]
		public float Aluminium;
		[Column("5")]
		public float Silicon;
		[Column("6")]
		public float Potassium;
		[Column("7")]
		public float Calcium;
		[Column("8")]
		public float Barium;
		[Column("9")]
		public float Iron;
		[Column("10", "Label")]
		public string Type;
	}

	public class GlassTypePrediction : MultiClassificationPrediction { }
}
