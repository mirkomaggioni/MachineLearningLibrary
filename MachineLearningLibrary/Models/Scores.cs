using Microsoft.ML.Runtime.Api;

namespace MachineLearningLibrary.Models
{
	public class ScoreLabel
	{
		public string Label { get; set; }
		public float Score { get; set; }
	}

	public class RegressionPrediction
	{
		[ColumnName("Score")]
		public float Score;
	}

	public class BinaryClassificationPrediction
	{
		[ColumnName("PredictedLabel")]
		public bool PredictedLabel;
	}

	public class MultiClassificationPrediction
	{
		[ColumnName("PredictedLabel")]
		public string PredictedLabel;
		[ColumnName("Score")]
		public float[] Scores;
	}
}
