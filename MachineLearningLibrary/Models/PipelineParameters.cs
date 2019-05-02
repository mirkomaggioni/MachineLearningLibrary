using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Text;
using System.Collections.Generic;

namespace MachineLearningLibrary.Models
{
	public class PipelineParameters<T> where T : class
	{
		public readonly MLContext MlContext;
		public readonly IDataView DataView;
		public readonly TextFeaturizingEstimator TextFeaturizingEstimator;
		//private readonly string[] _dictionarizedLabels;
		//private string _predictedColumn;

		//public PipelineParameters(string dataPath, char separator, string predictedColumn = null, string[] alphanumericColumns = null, string[] dictionarizedLabels = null, IEnumerable<string> concatenatedColumns = null)
		public PipelineParameters(string dataPath, char separator, string[] alphanumericColumns = null, IEnumerable<string> concatenatedColumns = null)
		{
			MlContext = new MLContext();
			DataView = MlContext.Data.LoadFromTextFile<T>(dataPath, separator, hasHeader: false);
			TextFeaturizingEstimator = MlContext.Transforms.Text.FeaturizeText(DefaultColumnNames.Features, concatenatedColumns, null);

			foreach (var alphanumericColumn in alphanumericColumns)
				MlContext.Transforms.Categorical.OneHotEncoding(alphanumericColumn);

			//_predictedColumn = predictedColumn;
			//_dictionarizedLabels = dictionarizedLabels;
		}

		//public void SetupTrainerEstimator<TTransformer, TModel>(ITrainerEstimator<TTransformer, TModel> trainerEstimator)
		//	where TTransformer : class, ITransformer
		//	//ISingleFeaturePredictionTransformer<TModel>
		//	where TModel : class
		//{
		//	EstimatorChain<ITransformer> EsstimatorChain = _textFeaturizingEstimator.Append(trainerEstimator);

		//	if (EstimatorChain == null)
		//		EstimatorChain = _textFeaturizingEstimator.Append(trainerEstimator);
		//}

		//public PredictedLabelColumnOriginalValueConverter PredictedLabelColumnOriginalValueConverter => !string.IsNullOrEmpty(_predictedColumn) ? new PredictedLabelColumnOriginalValueConverter { PredictedLabelColumn = _predictedColumn } : null;
		//public Dictionarizer Dictionarizer => _dictionarizedLabels != null ? new Dictionarizer(_dictionarizedLabels) : null;
	}
}
