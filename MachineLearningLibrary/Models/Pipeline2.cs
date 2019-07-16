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
	public class Pipeline2<T>
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
		private readonly AlgorithmType _algorithmType;
		private readonly (string, bool) _predictedColumn;
		private readonly string[] _alphanumericColumns;

		public Pipeline2(string dataPath, char separator, AlgorithmType algorithmType, (string, bool) predictedColumn, string[] concatenatedColumns, string[] alphanumericColumns = null)
		{
			MlContext = new MLContext();
			DataView = MlContext.Data.LoadFromTextFile<T>(dataPath, separator, hasHeader: false);
			_modelsRootPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\Models";
			Directory.CreateDirectory(_modelsRootPath);
			_algorithmType = algorithmType;
			_predictedColumn = predictedColumn;
			_concatenatedColumns = concatenatedColumns;
			_alphanumericColumns = alphanumericColumns;
		}

		public void BuildModel()
		{
			var keyConversion = _predictedColumn.Item2 ? MlContext.Transforms.Conversion.MapValueToKey(_predictedColumn.Item1) : null;
			var keyColumn = MlContext.Transforms.CopyColumns("Label", _predictedColumn.Item1);
			var featureColumn = MlContext.Transforms.Concatenate("Features", _concatenatedColumns);
			var defaultPipeline = _predictedColumn.Item2 ? keyConversion.Append(keyColumn).Append(featureColumn) : keyColumn.Append(featureColumn);

			EstimatorChain<OneHotEncodingTransformer> oneHotEncodingChain = null;
			if (_alphanumericColumns != null)
			{
				for (int i = 0; i < _alphanumericColumns.Length; i++)
				{
					if (oneHotEncodingChain == null)
					{
						oneHotEncodingChain = defaultPipeline.Append(MlContext.Transforms.Categorical.OneHotEncoding(_alphanumericColumns[i]));
					}
					else
					{
						oneHotEncodingChain = oneHotEncodingChain.Append(MlContext.Transforms.Categorical.OneHotEncoding(_alphanumericColumns[i]));
					}
				}
			}

			_model = GetModel(_algorithmType, defaultPipeline, oneHotEncodingChain);
			SaveModel(_model);
		}

		public void SaveModel(ITransformer _model)
		{
			var modelPath = $@"{_modelsRootPath}\{Guid.NewGuid()}.zip";

			using (var fileStream = new FileStream(modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
			{
				MlContext.Model.Save(_model, DataView.Schema, fileStream);
			}
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

		private ITransformer GetModel(AlgorithmType algorithmType, EstimatorChain<ColumnConcatenatingTransformer> pipeline, EstimatorChain<OneHotEncodingTransformer> oneHotEncodingChain)
		{
			switch (algorithmType)
			{
				case AlgorithmType.StochasticDualCoordinateAscentRegressor:
					return pipeline.Append(MlContext.Regression.Trainers.Sdca()).Fit(DataView);
				case AlgorithmType.FastTreeRegressor:
					return pipeline.Append(MlContext.Regression.Trainers.FastTree()).Fit(DataView);
				case AlgorithmType.FastTreeTweedieRegressor:
					return pipeline.Append(MlContext.Regression.Trainers.FastTreeTweedie()).Fit(DataView);
				case AlgorithmType.FastForestRegressor:
					return pipeline.Append(MlContext.Regression.Trainers.FastForest()).Fit(DataView);
				case AlgorithmType.OnlineGradientDescentRegressor:
					return pipeline.Append(MlContext.Regression.Trainers.OnlineGradientDescent()).Fit(DataView);
				case AlgorithmType.PoissonRegressor:
					return pipeline.Append(MlContext.Regression.Trainers.LbfgsPoissonRegression()).Fit(DataView);
				case AlgorithmType.NaiveBayesMultiClassifier:
					return pipeline.Append(MlContext.MulticlassClassification.Trainers.NaiveBayes()).Fit(DataView);
				case AlgorithmType.StochasticDualCoordinateAscentMultiClassifier:
					return pipeline.Append(MlContext.MulticlassClassification.Trainers.SdcaNonCalibrated()).Fit(DataView);
				case AlgorithmType.FastForestBinaryClassifier:
					return pipeline.Append(MlContext.BinaryClassification.Trainers.FastForest(numberOfTrees: 3000)).Fit(DataView);
				case AlgorithmType.AveragedPerceptronBinaryClassifier:
					return pipeline.Append(MlContext.BinaryClassification.Trainers.AveragedPerceptron()).Fit(DataView);
				case AlgorithmType.FastTreeBinaryClassifier:
					return pipeline.Append(MlContext.BinaryClassification.Trainers.FastTree()).Fit(DataView);
				case AlgorithmType.FieldAwareFactorizationMachineBinaryClassifier:
					return pipeline.Append(MlContext.BinaryClassification.Trainers.FieldAwareFactorizationMachine(_concatenatedColumns)).Fit(DataView);
				case AlgorithmType.LinearSvmBinaryClassifier:
					return pipeline.Append(MlContext.BinaryClassification.Trainers.LinearSvm()).Fit(DataView);
				case AlgorithmType.StochasticDualCoordinateAscentBinaryClassifier:
					return pipeline.Append(MlContext.BinaryClassification.Trainers.SdcaLogisticRegression()).Fit(DataView);
				case AlgorithmType.StochasticGradientDescentBinaryClassifier:
					return pipeline.Append(MlContext.BinaryClassification.Trainers.SgdNonCalibrated()).Fit(DataView);
				case AlgorithmType.LbfgsMultiClassifier:
					return pipeline.Append(MlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy()).Fit(DataView);
				case AlgorithmType.GamBinaryClassifier:
					return pipeline.Append(MlContext.BinaryClassification.Trainers.Gam()).Fit(DataView);
				case AlgorithmType.LbfgsBinaryClassifier:
					return pipeline.Append(MlContext.BinaryClassification.Trainers.LbfgsLogisticRegression()).Fit(DataView);
				default:
					return null;
			}
		}
	}
}
