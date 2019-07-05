using MachineLearningLibrary.Models;
using System.IO;
using System.Reflection;
using Microsoft.ML;

namespace MachineLearningLibrary.Services
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

	public class PredictionService
	{
		private readonly string _modelsRootPath;

		public PredictionService()
		{
			_modelsRootPath = $@"{Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location)}\Models";
			Directory.CreateDirectory(_modelsRootPath);
		}

		public TPredictionModel PredictScore<T, TPredictionModel>(T data, Pipeline<T> pipelineParameters, ITransformer model) 
			where T : class
			where TPredictionModel : class, IPredictionModel, new()
		{
			var predictionEngine = pipelineParameters.MlContext.Model.CreatePredictionEngine<T, TPredictionModel>(model);
			return predictionEngine.Predict(data);
		}
	}
}
