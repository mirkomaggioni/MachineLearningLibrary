using Microsoft.ML.Data;

namespace MachineLearningLibrary.Models
{
	public class ScoreLabel
	{
		public string Label { get; set; }
		public float Score { get; set; }
	}

	public class RegressionPrediction : IPredictionModel<float>
	{
		[ColumnName("Score")]
		public float Score;
	}

	public class BinaryClassificationPrediction : IPredictionModel<bool>
	{
		[ColumnName("PredictedLabel")]
		public bool PredictedLabel;
		public float Probability { get; set; }
		public float Score { get; set; }
	}

	public class MultiClassificationPrediction<T> : IPredictionModel<T>
	{
		[ColumnName("PredictedLabel")]
		public T PredictedLabel;
		[ColumnName("Score")]
		public float[] Scores;
	}
}
