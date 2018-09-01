using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace MachineLearningLibrary.Models
{
	public class PipelineParameters<T> where T : class
	{
		private readonly string[] _alphanumericColumns;
		private readonly string[] _dictionarizedLabels;
		private readonly string[] _concatenatedColumns;
		private string _predictedColumn;

		public PipelineParameters(string dataPath, char separator, string predictedColumn = null, string[] alphanumericColumns = null, string[] dictionarizedLabels = null, string[] concatenatedColumns = null)
		{
			TextLoader = new TextLoader(dataPath).CreateFrom<T>(separator: separator);
			_predictedColumn = predictedColumn;
			_alphanumericColumns = alphanumericColumns;
			_dictionarizedLabels = dictionarizedLabels;
			_concatenatedColumns = concatenatedColumns;
		}

		public TextLoader TextLoader { get; }
		public PredictedLabelColumnOriginalValueConverter PredictedLabelColumnOriginalValueConverter => !string.IsNullOrEmpty(_predictedColumn) ? new PredictedLabelColumnOriginalValueConverter { PredictedLabelColumn = _predictedColumn } : null;
		public Dictionarizer Dictionarizer => _dictionarizedLabels != null ? new Dictionarizer(_dictionarizedLabels) : null;
		public ColumnConcatenator ColumnConcatenator => _concatenatedColumns != null ? new ColumnConcatenator("Features", _concatenatedColumns) : null;
		public CategoricalOneHotVectorizer CategoricalOneHotVectorizer => _alphanumericColumns != null ? new CategoricalOneHotVectorizer(_alphanumericColumns) : null;
	}
}
