using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace MachineLearningLibrary.Models
{
	public class PipelineParameters<T> where T : class
	{
		public readonly MLContext MlContext;
		public readonly IDataView DataView;
		public readonly EstimatorChain<ColumnConcatenatingTransformer> Chain;
		public readonly string[] ConcatenatedColumns;

		public PipelineParameters(string dataPath, char separator, (string, bool) predictedColumn, string[] concatenatedColumns, string[] alphanumericColumns)
		{
			MlContext = new MLContext();
			DataView = MlContext.Data.LoadFromTextFile<T>(dataPath, separator, hasHeader: false);
			dynamic chain = null;

			if (predictedColumn.Item2)
			{
				chain = MlContext.Transforms.CopyColumns("Label", predictedColumn.Item1);
			}
			else
			{
				chain = MlContext.Transforms.Conversion.MapValueToKey(predictedColumn.Item1).Append(MlContext.Transforms.CopyColumns("Label", predictedColumn.Item1));
			}

			EstimatorChain<OneHotEncodingTransformer> OneHotEncodingEstimator = null;

			if (alphanumericColumns != null)
			{
				//MlContext.Transforms.Text.FeaturizeText(DefaultColumnNames.Features, alphanumericColumns, null);
				foreach (var alphanumericColumn in alphanumericColumns)
				{
					if (OneHotEncodingEstimator == null)
					{
						OneHotEncodingEstimator = chain.Append(MlContext.Transforms.Categorical.OneHotEncoding(alphanumericColumn));
					}
					else
					{
						OneHotEncodingEstimator = OneHotEncodingEstimator.Append(MlContext.Transforms.Categorical.OneHotEncoding(alphanumericColumn));
					}
				}

				Chain = OneHotEncodingEstimator.Append(MlContext.Transforms.Concatenate("Features", concatenatedColumns));
			}
			else
			{
				Chain = chain.Append(MlContext.Transforms.Concatenate("Features", concatenatedColumns));
			}

			ConcatenatedColumns = concatenatedColumns;
		}
	}
}
