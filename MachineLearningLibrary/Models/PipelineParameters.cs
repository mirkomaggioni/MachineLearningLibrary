using Microsoft.ML;
using Microsoft.ML.Runtime.Data;
using System;
using System.Collections.Generic;
using static Microsoft.ML.Runtime.Data.TextLoader;

namespace MachineLearningLibrary.Models
{
	public class PipelineParameters<T> where T : class
	{
		private readonly TextLoader _reader;
		private string _predictedColumn;
		private readonly string _dataPath;

		public PipelineParameters(string dataPath, string separator, bool hasHeader, string predictedColumn = null, Tuple<string, DataKind, int, bool, bool>[] columns = null, TrainContextBase trainContextBase = null)
		{
			var mlColumns = new List<Column>();
			var mlFeatureColumns  = new List<Range>();
			var mlTargetColumns  = new List<Range>();

			foreach (var column in columns)
			{
				mlColumns.Add(new Column(column.Item1, column.Item2, column.Item3));
				if (column.Item4)
				{
					mlFeatureColumns.Add(new Range(column.Item3));
				}
				if (column.Item5)
				{
					mlTargetColumns.Add(new Range(column.Item3));
				}
			}

			mlColumns.Add(new Column("FeatureVector", DataKind.R4, mlFeatureColumns.ToArray()));
			mlColumns.Add(new Column("Target", DataKind.R4, mlTargetColumns.ToArray()));

			_reader = new TextLoader(new LocalEnvironment(), new Arguments
			{
				Column = mlColumns.ToArray(),
				HasHeader = hasHeader,
				Separator = separator
			});

			_predictedColumn = predictedColumn;
            TrainContextBase = trainContextBase;
			_dataPath = dataPath;
		}

		public IDataView Data => _reader.Read(new MultiFileSource(_dataPath));
		public TrainContextBase TrainContextBase { get; }
	}
}
