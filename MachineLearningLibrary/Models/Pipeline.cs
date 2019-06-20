using MachineLearningLibrary.Interfaces;
using MachineLearningLibrary.Services;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace MachineLearningLibrary.Models
{
	public class Pipeline<T> : IPredictedColumnPipeline, IAlphanumericColumnsConversionPipeline, IConcatenateColumns, ITrain where T : class
	{
		public readonly MLContext MlContext;
		public readonly IDataView DataView;

		private ValueToKeyMappingEstimator valueToKeyMappingEstimator;
		private ColumnCopyingEstimator columnCopyingEstimator;
		private EstimatorChain<ColumnCopyingTransformer> estimatorChainCopy;
		private EstimatorChain<OneHotEncodingTransformer> estimatorChainEncoding;
		private EstimatorChain<ColumnConcatenatingTransformer> estimatorChainTransformer;

		public Pipeline(string dataPath, char separator)
		{
			MlContext = new MLContext();
			DataView = MlContext.Data.LoadFromTextFile<T>(dataPath, separator, hasHeader: false);
		}

		public IAlphanumericColumnsConversionPipeline ConvertAlphanumericKeyColumn(string column)
		{
			valueToKeyMappingEstimator = MlContext.Transforms.Conversion.MapValueToKey(column);
			return this;
		}

		public IAlphanumericColumnsConversionPipeline CopyColumn(string outputColumnName, string inputColumnName)
		{
			if (valueToKeyMappingEstimator == null)
			{
				columnCopyingEstimator = MlContext.Transforms.CopyColumns(outputColumnName, inputColumnName);
			}
			else if(columnCopyingEstimator != null)
			{
				estimatorChainCopy = columnCopyingEstimator.Append(MlContext.Transforms.CopyColumns(outputColumnName, inputColumnName));
			}
			else
			{
				estimatorChainCopy = valueToKeyMappingEstimator.Append(MlContext.Transforms.CopyColumns(outputColumnName, inputColumnName));
			}

			return this;
		}

		public IConcatenateColumns ConvertAlphanumericColumns(string[] columns)
		{
			for (int i = 0; i < columns.Length; i++)
			{
				if (i == 0)
				{
					estimatorChainEncoding = estimatorChainCopy != null 
						? estimatorChainCopy.Append(MlContext.Transforms.Categorical.OneHotEncoding(columns[i]))
						: columnCopyingEstimator.Append(MlContext.Transforms.Categorical.OneHotEncoding(columns[i]));
				}
				else
				{
					estimatorChainEncoding = estimatorChainEncoding.Append(MlContext.Transforms.Categorical.OneHotEncoding(columns[i]));
				}
			}

			return this;
		}

		public ITrain ConcatenateColumns(string[] columns)
		{
			if (estimatorChainEncoding != null)
			{
				estimatorChainTransformer = estimatorChainEncoding.Append(MlContext.Transforms.Concatenate("Features", columns));
			}
			else
			{
				estimatorChainTransformer = estimatorChainCopy!= null ? estimatorChainCopy.Append(MlContext.Transforms.Concatenate("Features", columns)) : columnCopyingEstimator.Append(MlContext.Transforms.Concatenate("Features", columns));
			}

			return this;
		}

		public void Train(AlgorithmType algorithmType)
		{
			throw new System.NotImplementedException();
		}
	}
}
