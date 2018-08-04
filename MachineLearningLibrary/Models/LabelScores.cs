using Microsoft.ML.Runtime.Api;

namespace MachineLearningLibrary.Models
{
	public class LabelScore
	{
		public string Label { get; set; }
		public float Score { get; set; }
	}

	public class LabelsScores
	{
		[ColumnName("PredictedLabel")]
		public string PredictedTypes;
		[ColumnName("Score")]
		public float[] Score;
		public LabelScore[] Scores { get; set; }
	}
}
