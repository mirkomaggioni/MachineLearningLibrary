using Microsoft.ML.Data;

namespace MachineLearningLibrary.Models
{
	public class ScoreLabel
	{
		public string Label { get; set; }
		public float Score { get; set; }
	}

	public class RegressionPrediction : IPredictionModel
	{
		[ColumnName("Score")]
		public float Score;
	}

	public class BinaryClassificationPrediction : IPredictionModel
	{
		[ColumnName("PredictedLabel")]
		public bool PredictedLabel;
		public float Probability { get; set; }
		public float Score { get; set; }
	}

	public class MultiClassificationPrediction : IPredictionModel
	{
		[ColumnName("PredictedLabel")]
		public string PredictedLabel;
		[ColumnName("Score")]
		public float[] Scores;
	}
}
