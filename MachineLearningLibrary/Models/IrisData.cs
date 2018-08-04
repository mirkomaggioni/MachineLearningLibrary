using Microsoft.ML.Runtime.Api;

namespace MachineLearningLibrary.Models
{
	public class IrisData
	{
		[Column("0")]
		public float SepalLength;

		[Column("1")]
		public float SepalWidth;

		[Column("2")]
		public float PetalLength;

		[Column("3")]
		public float PetalWidth;

		[Column("4", "Label")]
		public string Type;
	}

	public class IrisTypePrediction : LabelsScores {}
}
