using Microsoft.Data.DataView;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections.Generic;
using System.Linq;

namespace MachineLearningLibrary.Models
{
	public class PipelineParameters<T> where T : class
	{
		public readonly MLContext mlContext;
		public readonly IDataView dataView;

		private readonly string[] _dictionarizedLabels;
		private string _predictedColumn;

		public PipelineParameters(string dataPath, char separator, string predictedColumn = null, string[] alphanumericColumns = null, string[] dictionarizedLabels = null, IEnumerable<string> concatenatedColumns = null)
		{
			mlContext = new MLContext();
			dataView = mlContext.Data.LoadFromTextFile<T>(dataPath, separator, hasHeader: false);
			mlContext.Transforms.Text.FeaturizeText(DefaultColumnNames.Features, concatenatedColumns, null);

			foreach (var alphanumericColumn in alphanumericColumns)
				mlContext.Transforms.Categorical.OneHotEncoding(alphanumericColumn);

			_predictedColumn = predictedColumn;
			_dictionarizedLabels = dictionarizedLabels;
		}

		public PredictedLabelColumnOriginalValueConverter PredictedLabelColumnOriginalValueConverter => !string.IsNullOrEmpty(_predictedColumn) ? new PredictedLabelColumnOriginalValueConverter { PredictedLabelColumn = _predictedColumn } : null;
		public Dictionarizer Dictionarizer => _dictionarizedLabels != null ? new Dictionarizer(_dictionarizedLabels) : null;
	}
}
