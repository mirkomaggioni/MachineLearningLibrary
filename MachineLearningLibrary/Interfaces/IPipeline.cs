using MachineLearningLibrary.Services;

namespace MachineLearningLibrary.Interfaces
{
	public interface IPredictedColumnPipeline
	{
		IAlphanumericColumnsConversionPipeline ConvertAlphanumericKeyColumn(string column);
		IAlphanumericColumnsConversionPipeline CopyColumn(string outputColumnName, string inputColumnName);
	}

	public interface IAlphanumericColumnsConversionPipeline
	{
		IAlphanumericColumnsConversionPipeline CopyColumn(string outputColumnName, string inputColumnName);
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
