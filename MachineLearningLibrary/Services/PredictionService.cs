using MachineLearningLibrary.Models;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System.IO;
using System.Reflection;

namespace MachineLearningLibrary.Services
{
	public class PredictionService
	{
		public CarPricePrediction PricePrediction(Car car)
		{
			var pipeline = new LearningPipeline();
			var dir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
			pipeline.Add(new TextLoader($@"{dir}\cars.txt").CreateFrom<Car>(separator: ';'));
			pipeline.Add(new ColumnCopier(("Price", "Label")));
			pipeline.Add(new CategoricalOneHotVectorizer("Manufacturer", "Color", "Year"));
			pipeline.Add(new ColumnConcatenator("Features", "Manufacturer", "Color", "Year"));
			pipeline.Add(new FastTreeRegressor());

			var model = pipeline.Train<Car, CarPricePrediction>();
			return model.Predict(car);
		}
	}
}
