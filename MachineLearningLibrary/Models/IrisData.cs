using Microsoft.ML.Data;

namespace MachineLearningLibrary.Models
{
	public class IrisData
	{
		[LoadColumn(0)]
		public float SepalLength;

		[LoadColumn(1)]
		public float SepalWidth;

		[LoadColumn(2)]
		public float PetalLength;

		[LoadColumn(3)]
		public float PetalWidth;

		[LoadColumn(4), ColumnName("Label")]
		public string Type;
	}

	public class IrisTypePrediction : MultiClassificationPrediction {}
}
