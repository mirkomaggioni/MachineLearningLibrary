using Microsoft.ML.Runtime.Api;

namespace MachineLearningLibrary.Models
{
	public class MushroomData
	{
		[Column("0", "Label")]
		public bool Edible;
		[Column("1")]
		public string CapShape;
		[Column("2")]
		public string CapSurface;
		[Column("3")]
		public string CapColor;
		[Column("4")]
		public string Bruises;
		[Column("5")]
		public string Odor;
		[Column("6")]
		public string GillAttachment;
		[Column("7")]
		public string GillSpacing;
		[Column("8")]
		public string GillSize;
		[Column("9")]
		public string GillColor;
		[Column("10")]
		public string StalkShape;
		[Column("11")]
		public string StalkRoot;
		[Column("12")]
		public string StalkSurfaceAboveRing;
		[Column("13")]
		public string StalkSurfaceBelowRing;
		[Column("14")]
		public string StalkColorAboveRing;
		[Column("15")]
		public string StalkColorBelowRing;
		[Column("16")]
		public string VeilType;
		[Column("17")]
		public string VeilColor;
		[Column("18")]
		public string RingNumber;
		[Column("19")]
		public string RingType;
		[Column("20")]
		public string SporePrintColor;
		[Column("21")]
		public string Population;
		[Column("22")]
		public string Habitat;
	}

	public class MushroomEdiblePrediction : BinaryClassificationPrediction { }
}
