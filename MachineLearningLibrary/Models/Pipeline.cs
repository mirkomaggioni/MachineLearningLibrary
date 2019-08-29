using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.IO;
using System.Reflection;

namespace MachineLearningLibrary.Models
{
	public enum AlgorithmType
	{
		StochasticDualCoordinateAscentRegressor,
		FastTreeRegressor,
		FastTreeTweedieRegressor,
		FastForestRegressor,
		OnlineGradientDescentRegressor,
		PoissonRegressor,
		NaiveBayesMultiClassifier,
		StochasticDualCoordinateAscentMultiClassifier,
		FastForestBinaryClassifier,
		AveragedPerceptronBinaryClassifier,
		FastTreeBinaryClassifier,
		FieldAwareFactorizationMachineBinaryClassifier,
		LinearSvmBinaryClassifier,
		StochasticDualCoordinateAscentBinaryClassifier,
		StochasticGradientDescentBinaryClassifier,
		LbfgsMultiClassifier,
		GamBinaryClassifier,
		LbfgsBinaryClassifier
	}

	public class Pipeline<T>
	{
		public readonly MLContext MlContext;
		public readonly IDataView DataView;

		private ITransformer _model;
		private string[] _concatenatedColumns;
		private string _modelsRootPath;
		private readonly AlgorithmType? _algorithmType;
		private readonly PredictedColumn _predictedColumn;
		private readonly string[] _alphanumericColumns;
		private readonly string _featureColumn = "Features";
		private EstimatorChain<TransformerChain<ColumnConcatenatingTransformer>> _transformerChain = null;
		private EstimatorChain<ColumnConcatenatingTransformer> _estimatorChain = null;

		public Pipeline(string dataPath, char separator, AlgorithmType? algorithmType = null, PredictedColumn predictedColumn = null, string[] concatenatedColumns = null, string[] alphanumericColumns = null)
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

		public void BuildPipeline()
		{
			if (_predictedColumn == null)
				throw new ArgumentNullException(nameof(_predictedColumn));
			if (_algorithmType == null)
				throw new ArgumentNullException(nameof(_algorithmType));

			var keyMap = _predictedColumn.IsAlphanumeric ? MlContext.Transforms.Conversion.MapValueToKey(_predictedColumn.ColumnName) : null;
			var keyConversion = _predictedColumn.DataKind != null ? MlContext.Transforms.Conversion.ConvertType(_predictedColumn.ColumnName, outputKind: _predictedColumn.DataKind.Value) : null;
			var keyColumn = MlContext.Transforms.CopyColumns("Label", _predictedColumn.ColumnName);

			if (_alphanumericColumns != null)
			{
				OneHotEncodingEstimator oneHotEncodingTransformer = null;
				EstimatorChain<OneHotEncodingTransformer> oneHotEncodingTransformerChain = null;
				if (_alphanumericColumns != null)
				{
					for (int i = 0; i < _alphanumericColumns.Length; i++)
					{
						if (oneHotEncodingTransformer == null)
						{
							oneHotEncodingTransformer = MlContext.Transforms.Categorical.OneHotEncoding(_alphanumericColumns[i]);
						}
						else if (oneHotEncodingTransformerChain == null)
						{
							oneHotEncodingTransformerChain = oneHotEncodingTransformer.Append(MlContext.Transforms.Categorical.OneHotEncoding(_alphanumericColumns[i]));
						}
						else
						{
							oneHotEncodingTransformerChain = oneHotEncodingTransformerChain.Append(MlContext.Transforms.Categorical.OneHotEncoding(_alphanumericColumns[i]));
						}
					}
				}

				var columnConcatenatingTransformer = oneHotEncodingTransformerChain?.Append(MlContext.Transforms.Concatenate(_featureColumn, _concatenatedColumns)) ??
														oneHotEncodingTransformer.Append(MlContext.Transforms.Concatenate(_featureColumn, _concatenatedColumns));
				_transformerChain = _predictedColumn.IsAlphanumeric ?
												keyMap.Append(keyColumn).Append(columnConcatenatingTransformer) :
												_predictedColumn.DataKind != null ? keyConversion.Append(keyColumn).Append(columnConcatenatingTransformer) : keyColumn.Append(columnConcatenatingTransformer);
			}
			else
			{
				var featureColumn = MlContext.Transforms.Concatenate(_featureColumn, _concatenatedColumns);
				_estimatorChain = _predictedColumn.IsAlphanumeric ? 
												keyMap.Append(keyColumn).Append(featureColumn) : 
												_predictedColumn.DataKind != null ? keyConversion.Append(keyColumn).Append(featureColumn) : keyColumn.Append(featureColumn);
			}
		}

		public void BuildModel()
		{
			if (_algorithmType != null)
			{
				_model = _transformerChain != null ? GetModel(_algorithmType.Value, _transformerChain) : GetModel(_algorithmType.Value, _estimatorChain);
			}
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
			return MlContext.BinaryClassification.EvaluateNonCalibrated(_model.Transform(dataView));
		}

		public ClusteringMetrics EvaluateClustering(IDataView dataView)
		{
			return MlContext.Clustering.Evaluate(_model.Transform(dataView));
		}

		public TPredictionModel PredictScore<TModel, TPredictionModel, TPredictionType>(TModel data)
			where TModel : class
			where TPredictionModel : class, IPredictionModel<TPredictionType>, new()
		{
			var predictionEngine = MlContext.Model.CreatePredictionEngine<TModel, TPredictionModel>(_model);
			return predictionEngine.Predict(data);
		}

		private ITransformer GetModel(AlgorithmType algorithmType, EstimatorChain<ColumnConcatenatingTransformer> pipeline)
		{
			if (_predictedColumn.IsAlphanumeric)
				return pipeline.Append(GetAlgorithm(algorithmType)).Append(MlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel")).Fit(DataView);
			return pipeline.Append(GetAlgorithm(algorithmType)).Fit(DataView);
		}

		private ITransformer GetModel(AlgorithmType algorithmType, EstimatorChain<TransformerChain<ColumnConcatenatingTransformer>> pipeline)
		{
			if (_predictedColumn.IsAlphanumeric)
				return pipeline.Append(GetAlgorithm(algorithmType)).Append(MlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel")).Fit(DataView);
			return pipeline.Append(GetAlgorithm(algorithmType)).Fit(DataView);
		}

		private IEstimator<ITransformer> GetAlgorithm(AlgorithmType algorithmType)
		{
			switch (algorithmType)
			{
				case AlgorithmType.StochasticDualCoordinateAscentRegressor:
					return MlContext.Regression.Trainers.Sdca();
				case AlgorithmType.FastTreeRegressor:
					return MlContext.Regression.Trainers.FastTree();
				case AlgorithmType.FastTreeTweedieRegressor:
					return MlContext.Regression.Trainers.FastTreeTweedie();
				case AlgorithmType.FastForestRegressor:
					return MlContext.Regression.Trainers.FastForest();
				case AlgorithmType.OnlineGradientDescentRegressor:
					return MlContext.Regression.Trainers.OnlineGradientDescent();
				case AlgorithmType.PoissonRegressor:
					return MlContext.Regression.Trainers.LbfgsPoissonRegression();
				case AlgorithmType.NaiveBayesMultiClassifier:
					return MlContext.MulticlassClassification.Trainers.NaiveBayes();
				case AlgorithmType.StochasticDualCoordinateAscentMultiClassifier:
					return MlContext.MulticlassClassification.Trainers.SdcaNonCalibrated();
				case AlgorithmType.FastForestBinaryClassifier:
					return MlContext.BinaryClassification.Trainers.FastForest(numberOfTrees: 3000);
				case AlgorithmType.AveragedPerceptronBinaryClassifier:
					return MlContext.BinaryClassification.Trainers.AveragedPerceptron();
				case AlgorithmType.FastTreeBinaryClassifier:
					return MlContext.BinaryClassification.Trainers.FastTree();
				case AlgorithmType.FieldAwareFactorizationMachineBinaryClassifier:
					return MlContext.BinaryClassification.Trainers.FieldAwareFactorizationMachine(_concatenatedColumns);
				case AlgorithmType.LinearSvmBinaryClassifier:
					return MlContext.BinaryClassification.Trainers.LinearSvm();
				case AlgorithmType.StochasticDualCoordinateAscentBinaryClassifier:
					return MlContext.BinaryClassification.Trainers.SdcaLogisticRegression();
				case AlgorithmType.StochasticGradientDescentBinaryClassifier:
					return MlContext.BinaryClassification.Trainers.SgdNonCalibrated();
				case AlgorithmType.LbfgsMultiClassifier:
					return MlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy();
				case AlgorithmType.GamBinaryClassifier:
					return MlContext.BinaryClassification.Trainers.Gam();
				case AlgorithmType.LbfgsBinaryClassifier:
					return MlContext.BinaryClassification.Trainers.LbfgsLogisticRegression();
				default:
					throw new ArgumentOutOfRangeException(nameof(algorithmType));
			}
		}
	}
}
