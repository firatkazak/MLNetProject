using Microsoft.ML;
using ImageClassification.TensorFlow.ImageDataStructures;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;

namespace ImageClassification.TensorFlow.ModelScorer;

//TensorFlow ile oluşturulmuş bir modelin kullanılması ve resim sınıflandırma işlemlerini gerçekleştiren sınıf.
public class TFModelScorer
{
    private readonly string dataLocation;
    private readonly string imagesFolder;
    private readonly string modelLocation;
    private readonly string labelsLocation;
    private readonly MLContext mlContext;
    private static string ImageReal = nameof(ImageReal);

    public TFModelScorer(string dataLocation, string imagesFolder, string modelLocation, string labelsLocation)
    {
        this.dataLocation = dataLocation;
        this.imagesFolder = imagesFolder;
        this.modelLocation = modelLocation;
        this.labelsLocation = labelsLocation;
        mlContext = new MLContext();
    }

    public struct ImageNetSettings
    {
        public const int imageHeight = 224;//Giriş görüntülerinin boyutları(Yükseklik).
        public const int imageWidth = 224;// Giriş görüntülerinin boyutları(Genişlik).
        public const float mean = 117;//Giriş görüntülerinin ortalama değeri.
        public const bool channelsLast = true;//Kanalların son sırada olup olmadığını belirtir.
    }

    public struct InceptionSettings
    {
        public const string inputTensorName = "input";//TensorFlow modelinin giriş tensörü adı.
        public const string outputTensorName = "softmax2";//TensorFlow modelinin çıkış tensörü adı.
    }

    public void Score()
    {
        //model'i yükler;
        PredictionEngine<ImageNetData, ImageNetPrediction> model = LoadModel(dataLocation, imagesFolder, modelLocation);
        //test verilerini kullanarak tahminler yapar ve sonuçları işler.
        ImageNetData[] predictions = PredictDataUsingModel(dataLocation, imagesFolder, labelsLocation, model).ToArray();

    }

    private PredictionEngine<ImageNetData, ImageNetPrediction> LoadModel(string dataLocation, string imagesFolder, string modelLocation)
    {
        ConsoleHelpers.ConsoleWriteHeader("Read model");
        Console.WriteLine($"Model location: {modelLocation}");
        Console.WriteLine($"Images folder: {imagesFolder}");
        Console.WriteLine($"Training file: {dataLocation}");
        Console.WriteLine($"Default parameters: image size=({ImageNetSettings.imageWidth},{ImageNetSettings.imageHeight}), image mean: {ImageNetSettings.mean}");

        //Veri dosyasını (dataLocation) yükler.
        IDataView data = mlContext.Data.LoadFromTextFile<ImageNetData>(dataLocation, hasHeader: true);

        //
        EstimatorChain<TensorFlowTransformer> pipeline = mlContext.Transforms
            .LoadImages(outputColumnName: "input", imageFolder: imagesFolder, inputColumnName: nameof(ImageNetData.ImagePath))//LoadImages metodu, belirtilen klasördeki görüntüleri yükler. outputColumnName parametresi, işlemin sonucunun hangi sütunda tutulacağını belirtir (burada "input"). imageFolder parametresi, görüntülerin bulunduğu klasör yolunu belirtir. inputColumnName parametresi, ImageNetData sınıfındaki görüntü dosya yolunu tutan özelliğin adını alır.
            .Append(mlContext.Transforms
            .ResizeImages(outputColumnName: "input", imageWidth: ImageNetSettings.imageWidth, imageHeight: ImageNetSettings.imageHeight, inputColumnName: "input"))//ResizeImages metodu, yüklenen görüntüleri belirtilen boyutlara yeniden boyutlandırır. outputColumnName parametresi, işlemin sonucunun hangi sütunda tutulacağını belirtir ("input" sütunu üzerine yazılır). imageWidth, imageHeight parametreleri, görüntülerin hedef boyutlarını belirtir. inputColumnName parametresi, önceki adımdan gelen görüntülerin bulunduğu sütunun adını alır ("input").
            .Append(mlContext.Transforms
            .ExtractPixels(outputColumnName: "input", interleavePixelColors: ImageNetSettings.channelsLast, offsetImage: ImageNetSettings.mean))//ExtractPixels metodu, görüntülerden pikselleri çıkarır ve ön işleme yapar. outputColumnName parametresi, işlemin sonucunun hangi sütunda tutulacağını belirtir ("input" sütunu üzerine yazılır). interleavePixelColors parametresi, piksellerin renk kanallarının sırasını belirtir (burada son sıra). offsetImage parametresi, piksellerden çıkarılacak ortalama değeri belirtir (burada ImageNetSettings.mean).
            .Append(mlContext.Model
            .LoadTensorFlowModel(modelLocation)//LoadTensorFlowModel metodu, TensorFlow modelini belirtilen konumdan yükler.
            .ScoreTensorFlowModel(outputColumnNames: new[] { "softmax2" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: true));//ScoreTensorFlowModel metodu, TensorFlow modelini skorlamak için yapılandırır: outputColumnNames parametresi, modelin çıkış tensörünün adını belirtir ("softmax2"). inputColumnNames parametresi, modelin giriş tensörünün adını belirtir ("input"). addBatchDimensionInput parametresi, giriş tensörüne bir toplu boyutu ekleyip eklemediğini belirtir (true olarak ayarlanmış).

        //Fit metodu, oluşturulan akışı (pipeline) ve veri setini (data) kullanarak bir dönüşüm yapar ve bir ITransformer nesnesi döndürür. Bu dönüştürülmüş model, tahmin yapmak için kullanılabilir hale gelir.
        ITransformer model = pipeline.Fit(data);

        //CreatePredictionEngine metodu, veriye dayalı olarak bir tahmin motoru oluşturur. model parametresi, tahmin motorunun kullanacağı ITransformer modelini alır. ImageNetData giriş türü ve ImageNetPrediction çıkış türü belirtilir.
        PredictionEngine<ImageNetData, ImageNetPrediction> predictionEngine = mlContext.Model.CreatePredictionEngine<ImageNetData, ImageNetPrediction>(model);

        //Sonucu döndürür.
        return predictionEngine;
    }

    protected IEnumerable<ImageNetData> PredictDataUsingModel(string testLocation, string imagesFolder, string labelsLocation, PredictionEngine<ImageNetData, ImageNetPrediction> model)
    {
        ConsoleHelpers.ConsoleWriteHeader("Classify images");
        Console.WriteLine($"Images folder: {imagesFolder}");
        Console.WriteLine($"Training file: {testLocation}");
        Console.WriteLine($"Labels file: {labelsLocation}");

        //ReadLabels metodu, labelsLocation parametresi ile belirtilen dosyadan etiketleri okur ve labels dizisine atar.
        string[] labels = ModelHelpers.ReadLabels(labelsLocation);
        //ReadFromCsv metodu, testLocation ve imagesFolder parametreleri ile belirtilen CSV dosyasından test verilerini okur ve testData koleksiyonuna atar.
        IEnumerable<ImageNetData> testData = ImageNetData.ReadFromCsv(testLocation, imagesFolder);

        //Her Bir Görüntü için Tahmin Yapma:  Predict metodu ile model, sample görüntüsü için tahminde bulunur ve 
        foreach (ImageNetData sample in testData)
        {//testData koleksiyonundaki her bir ImageNetData örneği(sample) için model tarafından tahmin yapılır.

            //PredictedLabels özellikleri bir olasılık dizisi (probs) olarak döner;
            float[] probs = model.Predict(sample).PredictedLabels;

            //ImageNetDataProbability nesnesi oluşturulur ve sample görüntüsünün ImagePath ve Label özellikleri bu nesneye kopyalanır.
            ImageNetDataProbability imageData = new ImageNetDataProbability()
            {
                ImagePath = sample.ImagePath,
                Label = sample.Label
            };
            //GetBestLabel metodu, labels dizisi ve probs olasılık dizisi ile çağrılarak en yüksek olasılığa sahip etiketi (PredictedLabel) ve bu olasılığı (Probability) belirler.
            (imageData.PredictedLabel, imageData.Probability) = ModelHelpers.GetBestLabel(labels, probs);
            //Sonuçları konsola yazdırır.
            imageData.ConsoleWrite();
            //yield return imageData; ifadesi, imageData nesnesini çağıran kod parçasına geri döndürür. Bu şekilde foreach döngüsü içindeki her bir iterasyonda bir ImageNetDataProbability nesnesi döndürülür.
            yield return imageData;
        }
    }
}
