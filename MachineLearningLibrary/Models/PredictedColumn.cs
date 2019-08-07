using Microsoft.ML.Data;

namespace MachineLearningLibrary.Models
{
	public class PredictedColumn
	{
		public PredictedColumn(string columnName, bool isAlphanumeric = false, DataKind? dataKind = null)
		{
			ColumnName = columnName;
			IsAlphanumeric = isAlphanumeric;
			DataKind = dataKind;
		}

		public string ColumnName { get; set; }
		public bool IsAlphanumeric { get; set; }
		public DataKind? DataKind { get; set; }
	}
}
