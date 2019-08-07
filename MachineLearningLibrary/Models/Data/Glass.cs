using Microsoft.ML.Data;

namespace MachineLearningLibrary.Models.Data
{
	public class Glass
	{
		[LoadColumn(0)]
		public float IdNumber;
		[LoadColumn(1)]
		public float RefractiveIndex;
		[LoadColumn(2)]
		public float Sodium;
		[LoadColumn(3)]
		public float Magnesium;
		[LoadColumn(4)]
		public float Aluminium;
		[LoadColumn(5)]
		public float Silicon;
		[LoadColumn(6)]
		public float Potassium;
		[LoadColumn(7)]
		public float Calcium;
		[LoadColumn(8)]
		public float Barium;
		[LoadColumn(9)]
		public float Iron;
		[LoadColumn(10)]
		public uint Type;
	}

	public class GlassTypePrediction : MultiClassificationPrediction<uint> { }
}
