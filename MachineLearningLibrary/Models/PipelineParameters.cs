using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace MachineLearningLibrary.Models
{
	public class PipelineParameters<T> where T : class
	{
		private readonly string[] _alphanumericColumns;
		private readonly string[] _dictionarizedLabels;
		private readonly string[] _concatenatedColumns;

		public PipelineParameters(string dataPath, char separator, string predictedColumn = null, string[] alphanumericColumns = null, string[] dictionarizedLabels = null, string[] concatenatedColumns = null)
		{
			TextLoader = new TextLoader(dataPath).CreateFrom<T>(separator: separator);
			PredictedColumn = predictedColumn;
			_alphanumericColumns = alphanumericColumns;
			_dictionarizedLabels = dictionarizedLabels;
			_concatenatedColumns = concatenatedColumns;
		}

		public TextLoader TextLoader { get; }
		public string LabelColumn { get; }
		public string PredictedColumn { get; }
		public PredictedLabelColumnOriginalValueConverter PredictedLabelColumnOriginalValueConverter => !string.IsNullOrEmpty(PredictedColumn) ? new PredictedLabelColumnOriginalValueConverter { PredictedLabelColumn = PredictedColumn } : null;
		public Dictionarizer Dictionarizer => _dictionarizedLabels != null ? new Dictionarizer(_dictionarizedLabels) : null;
		public ColumnConcatenator ColumnConcatenator => _concatenatedColumns != null ? new ColumnConcatenator("Features", _concatenatedColumns) : null;
		public CategoricalOneHotVectorizer CategoricalOneHotVectorizer => _alphanumericColumns != null ? new CategoricalOneHotVectorizer(_alphanumericColumns) : null;
	}
}
