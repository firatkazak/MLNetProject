using Common;
using ImageClassification.Train.ImageData;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System.Diagnostics;
using static Microsoft.ML.DataOperationsCatalog;

namespace ImageClassification.Train.Model;

public class ModelBuilder
{
    private readonly MLContext mlContext;
    private readonly string dataLocation;
    private readonly string imagesFolder;
    private readonly string inputTensorFlowModelFilePath;
    private readonly string outputMlNetModelFilePath;
    private static string LabelAsKey = nameof(LabelAsKey);
    private static string ImageReal = nameof(ImageReal);
    private static string PredictedLabelValue = nameof(PredictedLabelValue);

    public ModelBuilder(string inputModelLocation, string outputModelLocation)
    {
        this.inputTensorFlowModelFilePath = inputModelLocation;
        this.outputMlNetModelFilePath = outputModelLocation;
        mlContext = new MLContext(seed: 1);
    }

    //ImageSettingsForTFModel: TensorFlow modeli için gerekli olan bazı görüntü ayarlarını tanımlar: görüntü yüksekliği, genişliği, ortalama değeri ve ölçekleme faktörü gibi;
    private struct ImageSettingsForTFModel
    {
        public const int imageHeight = 299;
        public const int imageWidth = 299;
        public const float mean = 117;
        public const float scale = 1 / 255f;
        public const bool channelsLast = true;
    }

    //BuildAndTrain: Görüntü veri setini alır ve eğitim sürecini başlatır. Veri yükleme, karıştırma ve eğitim-test seti ayrımı gibi ön işlemleri gerçekleştirir.
    public void BuildAndTrain(IEnumerable<ImageData.ImageData> imageSet)
    {
        ConsoleHelpers.ConsoleWriteHeader("Read model");
        Console.WriteLine($"Model location: {inputTensorFlowModelFilePath}");
        Console.WriteLine($"Training file: {dataLocation}");

        IDataView fullImagesDataset = mlContext.Data.LoadFromEnumerable(imageSet);
        IDataView shuffledFullImagesDataset = mlContext.Data.ShuffleRows(fullImagesDataset);

        TrainTestData trainTestData = mlContext.Data.TrainTestSplit(shuffledFullImagesDataset, testFraction: 0.10);
        IDataView trainDataView = trainTestData.TrainSet;
        IDataView testDataView = trainTestData.TestSet;

        //Veri işleme işlemlerini içeren bir ML.NET boru hattı oluşturur. Burada, görüntüleri yüklemek, yeniden boyutlandırmak, pikselleri çıkarmak ve TensorFlow modelini kullanarak tahmin yapmak gibi işlemler yapılır.
        EstimatorChain<TensorFlowTransformer> dataProcessPipeline =
            mlContext.Transforms.Conversion
            .MapValueToKey(outputColumnName: LabelAsKey, inputColumnName: "Label")
            .Append(mlContext.Transforms
            .LoadImages(outputColumnName: "image_object", imageFolder: imagesFolder, inputColumnName: nameof(ImageData.ImageData.ImagePath)))
            .Append(mlContext.Transforms
            .ResizeImages(outputColumnName: "image_object_resized", imageWidth: ImageSettingsForTFModel.imageWidth, imageHeight: ImageSettingsForTFModel.imageHeight, inputColumnName: "image_object"))
            .Append(mlContext.Transforms.ExtractPixels(outputColumnName: "input", inputColumnName: "image_object_resized", interleavePixelColors: ImageSettingsForTFModel.channelsLast, offsetImage: ImageSettingsForTFModel.mean, scaleImage: ImageSettingsForTFModel.scale))
            .Append(mlContext.Model.LoadTensorFlowModel(inputTensorFlowModelFilePath)
            .ScoreTensorFlowModel(outputColumnNames: new[] { "InceptionV3/Predictions/Reshape" }, inputColumnNames: new[] { "input" }, addBatchDimensionInput: false));

        //LbfgsMaximumEntropyMulticlassTrainer ile çok sınıflı sınıflandırma eğitim algoritmasını tanımlar.
        LbfgsMaximumEntropyMulticlassTrainer trainer =
            mlContext.MulticlassClassification.Trainers.LbfgsMaximumEntropy(labelColumnName: LabelAsKey, featureColumnName: "InceptionV3/Predictions/Reshape");

        //Veri işleme boru hattına eğitim algoritmasını ekler ve tahmin sonuçlarını anahtarları gerçek değerlere dönüştürmek için dönüşüm ekler.
        EstimatorChain<KeyToValueMappingTransformer> trainingPipeline =
            dataProcessPipeline.Append(trainer).Append(mlContext.Transforms.Conversion.MapKeyToValue(PredictedLabelValue, "PredictedLabel"));

        //Stopwatch: Eğitim süresini ölçmek için kullanılır. Stopwatch.StartNew() ile süreç başlatılır.
        Stopwatch watch = Stopwatch.StartNew();

        ConsoleHelpers.ConsoleWriteHeader("Training the ML.NET classification model");
        //Fit: Belirtilen eğitim veri görünümü üzerinde modeli eğitir ve ITransformer tipinde eğitilen modeli döndürür.
        ITransformer model = trainingPipeline.Fit(trainDataView);
        //süreç durdurulur ve geçen süre elapsedMs değişkenine atanır.
        watch.Stop();
        long elapsedMs = watch.ElapsedMilliseconds;
        //Son olarak, eğitim süresi saniye cinsinden konsola yazdırılır.
        Console.WriteLine("Training with transfer learning took: " + (elapsedMs / 1000).ToString() + " seconds");

        ConsoleHelpers.ConsoleWriteHeader("Create Predictions and Evaluate the model quality");
        //model.Transform(testDataView) ile eğitilmiş model, test veri görünümü üzerinde tahminler oluşturur.
        IDataView predictionsDataView = model.Transform(testDataView);

        //predictionsDataView içindeki sütun adları alınarak loadedModelOutputColumnNames değişkenine atanır.
        IEnumerable<string> loadedModelOutputColumnNames = predictionsDataView.Schema.Where(col => !col.IsHidden).Select(col => col.Name);

        ConsoleHelpers.ConsoleWriteHeader("*** Showing all the predictions ***");

        //Tahmin edilen sonuçlar ImagePredictionEx tipinde bir liste olarak alınır ve konsola yazdırılır.
        List<ImagePredictionEx> predictions = mlContext.Data.CreateEnumerable<ImagePredictionEx>(predictionsDataView, false, true).ToList();
        predictions.ForEach(pred => ConsoleHelpers.ConsoleWriteImagePrediction(pred.ImagePath, pred.Label, pred.PredictedLabelValue, pred.Score.Max()));

        //mlContext.MulticlassClassification.Evaluate ile tahminlerin sınıflandırma metrikleri hesaplanır ve metrics değişkenine atanır.
        MulticlassClassificationCatalog classificationContext = mlContext.MulticlassClassification;
        ConsoleHelpers.ConsoleWriteHeader("Classification metrics");
        MulticlassClassificationMetrics metrics = classificationContext.Evaluate(predictionsDataView, labelColumnName: LabelAsKey, predictedLabelColumnName: "PredictedLabel");
        
        //Sınıflandırma metrikleri konsola yazdırılır;
        ConsoleHelper.PrintMultiClassClassificationMetrics(trainer.ToString(), metrics);
        ConsoleHelpers.ConsoleWriteHeader("Save model to local file");
        //DeleteAssets: Belirtilen dosya yolundaki varlıkları siler
        ModelHelpers.DeleteAssets(outputMlNetModelFilePath);

        //mlContext.Model.Save ile eğitilmiş model ve ilgili veri şeması belirtilen dosya yoluna kaydedilir.
        mlContext.Model.Save(model, predictionsDataView.Schema, outputMlNetModelFilePath);
        //Modelin başarıyla kaydedildiği mesajı konsola yazdırılır.
        Console.WriteLine($"Model saved: {outputMlNetModelFilePath}");
    }
}
