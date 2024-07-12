using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.Onnx;
using ObjectDetection.DataStructures;
using ObjectDetection.YoloParser;

namespace ObjectDetection;
class OnnxModelScorer
{
    private readonly string imagesFolder;//Kullanılacak ONNX modelinin dosya yolu.
    private readonly string modelLocation;//Nesne tespiti yapılacak görüntü dosyalarının dizini 
    private readonly MLContext mlContext;//ML.NET bağlamı, model yüklemek ve kullanmak için kullanılır.
    private IList<YoloBoundingBox> _boundingBoxes = new List<YoloBoundingBox>();//Nesne tespiti sonuçlarını tutmak için kullanılan bir liste.

    //Constructor metot;
    public OnnxModelScorer(string imagesFolder, string modelLocation, MLContext mlContext)
    {
        this.imagesFolder = imagesFolder;
        this.modelLocation = modelLocation;
        this.mlContext = mlContext;
    }

    //
    public struct ImageNetSettings
    {
        public const int imageHeight = 416;//Giriş görüntülerinin boyutları için sabitler.
        public const int imageWidth = 416;//Giriş görüntülerinin boyutları için sabitler.
    }

    //
    public struct TinyYoloModelSettings
    {
        public const string ModelInput = "image";//ModelInput ve ModelOutput: ONNX modelinin giriş ve çıkış sütun isimleri.
        public const string ModelOutput = "grid";//ModelInput ve ModelOutput: ONNX modelinin giriş ve çıkış sütun isimleri.
    }

    //LoadModel: ONNX modelini yüklemek için kullanılır. Görüntüleri yükler, yeniden boyutlandırır, pikselleri çıkarır ve ONNX modelini uygular.
    private ITransformer LoadModel(string modelLocation)
    {
        Console.WriteLine("Read model");
        Console.WriteLine($"Model location: {modelLocation}");
        Console.WriteLine($"Default parameters: image size=({ImageNetSettings.imageWidth},{ImageNetSettings.imageHeight})");

        IDataView data = mlContext.Data.LoadFromEnumerable(new List<ImageNetData>());

        EstimatorChain<OnnxTransformer> pipeline = mlContext.Transforms
            .LoadImages(outputColumnName: "image", imageFolder: "", inputColumnName: nameof(ImageNetData.ImagePath))
            .Append(mlContext.Transforms.ResizeImages(outputColumnName: "image", imageWidth: ImageNetSettings.imageWidth, imageHeight: ImageNetSettings.imageHeight, inputColumnName: "image"))
            .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "image"))
            .Append(mlContext.Transforms.ApplyOnnxModel(modelFile: modelLocation, outputColumnNames: new[] { TinyYoloModelSettings.ModelOutput }, inputColumnNames: new[] { TinyYoloModelSettings.ModelInput }));

        TransformerChain<OnnxTransformer> model = pipeline.Fit(data);

        return model;
    }

    //PredictDataUsingModel: Test verilerini kullanarak modeli değerlendirir ve sonuçları döndürür. Her bir görüntü için nesne tespiti olasılıklarını içeren bir dizi döner.
    private IEnumerable<float[]> PredictDataUsingModel(IDataView testData, ITransformer model)
    {
        Console.WriteLine($"Images location: {imagesFolder}");
        Console.WriteLine("");
        Console.WriteLine("=====Identify the objects in the images=====");
        Console.WriteLine("");

        IDataView scoredData = model.Transform(testData);

        IEnumerable<float[]> probabilities = scoredData.GetColumn<float[]>(TinyYoloModelSettings.ModelOutput);

        return probabilities;
    }

    //Score: Veri kümesini değerlendirir ve nesne tespiti sonuçlarını döndürür. Önce model yüklenir (LoadModel metoduyla), sonra test verileri üzerinde tahminleme yapılır (PredictDataUsingModel metoduyla).
    public IEnumerable<float[]> Score(IDataView data)
    {
        ITransformer model = LoadModel(modelLocation);

        return PredictDataUsingModel(data, model);
    }
}
//OnnxModelScorer: Bu sınıf, ML.NET ve ONNX kullanarak nesne tespiti yapmak için bir yapı sağlar. Giriş görüntülerini işler, ONNX modelini yükler, ve sonuçları çıkarır. Score metodunu çağırarak nesne tespiti sonuçlarına erişebilirsiniz.