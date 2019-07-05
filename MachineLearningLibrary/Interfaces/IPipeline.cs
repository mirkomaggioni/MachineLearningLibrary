using MachineLearningLibrary.Services;
using Microsoft.ML;
using Microsoft.ML.Data;

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
		ITrain ConcatenateColumns(string[] columns);
	}

	public interface IConcatenateColumns
	{
		ITrain ConcatenateColumns(string[] columns);
	}

	public interface ITrain
	{
		IPipelineTransformer Train(AlgorithmType algorithmType);
	}

	public interface IPipelineTransformer
	{
		RegressionMetrics EvaluateRegression(IDataView dataView);
		BinaryClassificationMetrics EvaluateBinaryClassification(IDataView dataView);
		ClusteringMetrics EvaluateClustering(IDataView dataView);
	}
}
