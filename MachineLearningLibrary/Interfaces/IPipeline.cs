using MachineLearningLibrary.Services;

namespace MachineLearningLibrary.Interfaces
{
	public interface IPredictedColumnPipeline
	{
		void ConvertAlphanumericKeyColumn(string column);
		IAlphanumericColumnsConversionPipeline CopyColumn(string outputColumnName, string inputColumnName);
	}

	public interface IAlphanumericColumnsConversionPipeline
	{
		IConcatenateColumns ConvertAlphanumericColumns(string[] columns);
	}

	public interface IConcatenateColumns
	{
		ITrain ConcatenateColumns(string[] columns);
	}

	public interface ITrain
	{
		void Train(AlgorithmType algorithmType);
	}
}
