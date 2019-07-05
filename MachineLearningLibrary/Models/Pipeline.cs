using MachineLearningLibrary.Interfaces;
using MachineLearningLibrary.Services;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.IO;
using System.Reflection;

namespace MachineLearningLibrary.Models
{
	public class Pipeline<T> : IPredictedColumnPipeline, IAlphanumericColumnsConversionPipeline, IConcatenateColumns, ITrain, IPipelineTransformer where T : class
	{
		public readonly MLContext MlContext;
		public readonly IDataView DataView;

		private ValueToKeyMappingEstimator valueToKeyMappingEstimator;
		private ColumnCopyingEstimator columnCopyingEstimator;
		private EstimatorChain<ColumnCopyingTransformer> estimatorChainCopy;
		private EstimatorChain<OneHotEncodingTransformer> estimatorChainEncoding;
		private EstimatorChain<ColumnConcatenatingTransformer> estimatorChainTransformer;
		private ITransformer _model;
		private string[] _concatenatedColumns;
		private string _modelsRootPath;

		public Pipeline(string dataPath, char separator)
		{
			MlContext = new MLContext();
			DataView = MlContext.Data.LoadFromTextFile<T>(dataPath, separator, hasHeader: false);
			_modelsRootPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\Models";
			Directory.CreateDirectory(_modelsRootPath);
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
			else if (columnCopyingEstimator != null)
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
			estimatorChainTransformer = estimatorChainEncoding != null ?
										estimatorChainEncoding.Append(MlContext.Transforms.Concatenate("Features", columns)) :
										estimatorChainCopy != null ? estimatorChainCopy.Append(MlContext.Transforms.Concatenate("Features", columns)) :
										columnCopyingEstimator.Append(MlContext.Transforms.Concatenate("Features", columns));
			_concatenatedColumns = columns;

			return this;
		}

		public IPipelineTransformer Train(AlgorithmType algorithmType)
		{
			var modelPath = $@"{_modelsRootPath}\{Guid.NewGuid()}.zip";
			_model = GetModel(algorithmType);

			using (var fileStream = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
			{
				MlContext.Model.Save(_model, DataView.Schema, fileStream);
			}
			return this;
		}

		public RegressionMetrics EvaluateRegression(IDataView dataView)
		{
			return MlContext.Regression.Evaluate(_model.Transform(dataView));
		}

		public BinaryClassificationMetrics EvaluateBinaryClassification(IDataView dataView)
		{
			return MlContext.BinaryClassification.Evaluate(_model.Transform(dataView));
		}

		public ClusteringMetrics EvaluateClustering(IDataView dataView)
		{
			return MlContext.Clustering.Evaluate(_model.Transform(dataView));
		}

		private ITransformer GetModel(AlgorithmType algorithmType)
		{
			switch (algorithmType)
			{
				case AlgorithmType.StochasticDualCoordinateAscentRegressor:
					return estimatorChainTransformer.Append(MlContext.Regression.Trainers.Sdca()).Fit(DataView);
				case AlgorithmType.FastTreeRegressor:
					return estimatorChainTransformer.Append(MlContext.Regression.Trainers.FastTree()).Fit(DataView);
				case AlgorithmType.FastTreeTweedieRegressor:
					return estimatorChainTransformer.Append(MlContext.Regression.Trainers.FastTreeTweedie()).Fit(DataView);
				case AlgorithmType.FastForestRegressor:
					return estimatorChainTransformer.Append(MlContext.Regression.Trainers.FastForest()).Fit(DataView);
				case AlgorithmType.OnlineGradientDescentRegressor:
					return estimatorChainTransformer.Append(MlContext.Regression.Trainers.OnlineGradientDescent()).Fit(DataView);
				case AlgorithmType.PoissonRegressor:
					return estimatorChainTransformer.Append(MlContext.Regression.Trainers.LbfgsPoissonRegression()).Fit(DataView);
				case AlgorithmType.NaiveBayesMultiClassifier:
					return estimatorChainTransformer.Append(MlContext.MulticlassClassification.Trainers.NaiveBayes()).Fit(DataView);
				case AlgorithmType.StochasticDualCoordinateAscentMultiClassifier:
					return estimatorChainTransformer.Append(MlContext.MulticlassClassification.Trainers.SdcaNonCalibrated()).Fit(DataView);
				case AlgorithmType.FastForestBinaryClassifier:
					return estimatorChainTransformer.Append(MlContext.BinaryClassification.Trainers.FastForest(numberOfTrees: 3000)).Fit(DataView);
				case AlgorithmType.AveragedPerceptronBinaryClassifier:
					return estimatorChainTransformer.Append(MlContext.BinaryClassification.Trainers.AveragedPerceptron()).Fit(DataView);
				case AlgorithmType.FastTreeBinaryClassifier:
					return estimatorChainTransformer.Append(MlContext.BinaryClassification.Trainers.FastTree()).Fit(DataView);
				case AlgorithmType.FieldAwareFactorizationMachineBinaryClassifier:
					return estimatorChainTransformer.Append(MlContext.BinaryClassification.Trainers.FieldAwareFactorizationMachine(_concatenatedColumns)).Fit(DataView);
				case AlgorithmType.LinearSvmBinaryClassifier:
					return estimatorChainTransformer.Append(MlContext.BinaryClassification.Trainers.LinearSvm()).Fit(DataView);
				case AlgorithmType.StochasticDualCoordinateAscentBinaryClassifier:
					return estimatorChainTransformer.Append(MlContext.BinaryClassification.Trainers.SdcaLogisticRegression()).Fit(DataView);
				case AlgorithmType.StochasticGradientDescentBinaryClassifier:
					return estimatorChainTransformer.Append(MlContext.BinaryClassification.Trainers.SgdNonCalibrated()).Fit(DataView);
				case AlgorithmType.LbfgsMultiClassifier:
					return estimatorChainTransformer.Append(MlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy()).Fit(DataView);
				case AlgorithmType.GamBinaryClassifier:
					return estimatorChainTransformer.Append(MlContext.BinaryClassification.Trainers.Gam()).Fit(DataView);
				case AlgorithmType.LbfgsBinaryClassifier:
					return estimatorChainTransformer.Append(MlContext.BinaryClassification.Trainers.LbfgsLogisticRegression()).Fit(DataView);
				default:
					return null;
			}
		}
	}
}
