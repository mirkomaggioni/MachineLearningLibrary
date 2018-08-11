using Microsoft.ML.Runtime.Api;

namespace MachineLearningLibrary.Models
{
	public class ScoreLabel
	{
		public string Label { get; set; }
		public float Score { get; set; }
	}

	public class SingleScore
	{
		[ColumnName("Score")]
		public float Score;
	}

	public class MultipleScores
	{
		[ColumnName("PredictedLabel")]
		public string PredictedLabel;
		[ColumnName("Score")]
		public float[] Score;
		public ScoreLabel[] Scores { get; set; }
	}
}
