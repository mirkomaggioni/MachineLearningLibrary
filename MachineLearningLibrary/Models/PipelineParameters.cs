using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace MachineLearningLibrary.Models
{
	public class PipelineParameters<T> where T : class
	{
		public readonly MLContext MlContext;
		public readonly IDataView DataView;
		//public readonly TextFeaturizingEstimator TextFeaturizingEstimator;
		public readonly EstimatorChain<ColumnConcatenatingTransformer> EstimatorChain;
		public readonly string[] ConcatenatedColumns;
		//private string _predictedColumn;

		//public PipelineParameters(string dataPath, char separator, string predictedColumn = null, string[] alphanumericColumns = null, string[] dictionarizedLabels = null, IEnumerable<string> concatenatedColumns = null)
		public PipelineParameters(string dataPath, char separator, string predictedColumn, string[] concatenatedColumns, string[] alphanumericColumns = null)
		{
			MlContext = new MLContext();
			DataView = MlContext.Data.LoadFromTextFile<T>(dataPath, separator, hasHeader: false);

			var columnCopyingEstimator = MlContext.Transforms.CopyColumns("Label", predictedColumn);
			//TextFeaturizingEstimator = MlContext.Transforms.Text.FeaturizeText(DefaultColumnNames.Features, alphanumericColumns, null);
			EstimatorChain<OneHotEncodingTransformer> estimatorChainColumnConcat = null;

			if (alphanumericColumns != null)
			{
				//MlContext.Transforms.Text.FeaturizeText(DefaultColumnNames.Features, alphanumericColumns, null);
				foreach (var alphanumericColumn in alphanumericColumns)
				{
					if (estimatorChainColumnConcat == null)
					{
						estimatorChainColumnConcat = columnCopyingEstimator.Append(MlContext.Transforms.Categorical.OneHotEncoding(alphanumericColumn));
					}
					else
					{
						estimatorChainColumnConcat = estimatorChainColumnConcat.Append(MlContext.Transforms.Categorical.OneHotEncoding(alphanumericColumn));
					}
				}

				EstimatorChain = estimatorChainColumnConcat.Append(MlContext.Transforms.Concatenate("Features", concatenatedColumns));
			}
			else
			{
				EstimatorChain = columnCopyingEstimator.Append(MlContext.Transforms.Concatenate("Features", concatenatedColumns));
			}

			ConcatenatedColumns = concatenatedColumns;
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
