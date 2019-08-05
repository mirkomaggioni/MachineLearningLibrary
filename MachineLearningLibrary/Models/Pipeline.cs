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
		private readonly (string, bool, DataKind?)? _predictedColumn;
		private readonly string[] _alphanumericColumns;
		private readonly string _featureColumn = "Features";

		public Pipeline(string dataPath, char separator, AlgorithmType? algorithmType = null, (string, bool, DataKind?)? predictedColumn = null, string[] concatenatedColumns = null, string[] alphanumericColumns = null)
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
			if (_predictedColumn == null)
				throw new ArgumentNullException(nameof(_predictedColumn));
			if (_algorithmType == null)
				throw new ArgumentNullException(nameof(_algorithmType));

			var keyMap = _predictedColumn.Value.Item2 ? MlContext.Transforms.Conversion.MapValueToKey(_predictedColumn.Value.Item1) : null;
			var keyConversion = _predictedColumn.Value.Item3 != null ? MlContext.Transforms.Conversion.ConvertType(_predictedColumn.Value.Item1, outputKind: _predictedColumn.Value.Item3.Value) : null;
			var keyColumn = MlContext.Transforms.CopyColumns("Label", _predictedColumn.Value.Item1);

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
				var defaultPipeline = _predictedColumn.Value.Item2 ?
												keyMap.Append(keyColumn).Append(columnConcatenatingTransformer) :
												_predictedColumn.Value.Item3 != null ? keyConversion.Append(keyColumn).Append(columnConcatenatingTransformer) : keyColumn.Append(columnConcatenatingTransformer);
				_model = GetModel(_algorithmType.Value, defaultPipeline);
			}
			else
			{
				var featureColumn = MlContext.Transforms.Concatenate(_featureColumn, _concatenatedColumns);
				var defaultPipeline = _predictedColumn.Value.Item2 ? 
												keyMap.Append(keyColumn).Append(featureColumn) : 
												_predictedColumn.Value.Item3 != null ? keyConversion.Append(keyColumn).Append(featureColumn) : keyColumn.Append(featureColumn);
				_model = GetModel(_algorithmType.Value, defaultPipeline);
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

		public TPredictionModel PredictScore<TModel, TPredictionModel>(TModel data)
			where TModel : class
			where TPredictionModel : class, IPredictionModel, new()
		{
			var predictionEngine = MlContext.Model.CreatePredictionEngine<TModel, TPredictionModel>(_model);
			return predictionEngine.Predict(data);
		}

		private ITransformer GetModel(AlgorithmType algorithmType, EstimatorChain<ColumnConcatenatingTransformer> pipeline)
		{
			return pipeline.Append(GetAlgorithm(algorithmType)).Fit(DataView);
		}

		private ITransformer GetModel(AlgorithmType algorithmType, EstimatorChain<TransformerChain<ColumnConcatenatingTransformer>> pipeline)
		{
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
