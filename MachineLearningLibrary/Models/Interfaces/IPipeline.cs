using Microsoft.ML;
using Microsoft.ML.Data;

namespace MachineLearningLibrary.Models.Interfaces
{
	public interface IPipeline
	{
		IPipelineChain BuildPipeline();
	}

	public interface IPipelineChain
	{
		IPipelineTransformer BuildModel();
	}

	public interface IPipelineTransformer
	{
		IPipelineModel SaveModel(ITransformer model);
	}

	public interface IPipelineModel
	{
		RegressionMetrics EvaluateRegression(IDataView dataView);
		BinaryClassificationMetrics EvaluateBinaryClassification(IDataView dataView);
		ClusteringMetrics EvaluateClustering(IDataView dataView);
		IDataView Transform();
		TPredictionModel PredictScore<TModel, TPredictionModel, TPredictionType>(TModel data) where TModel : class where TPredictionModel : class, IPredictionModel<TPredictionType>, new();
	}
}
