using Common;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using MulticlassClassificationIris.DataStructures;

string AppPath = Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);//Uygulamanın çalıştığı dizini alır.
string BaseDatasetsRelativePath = @"C:\Users\firat\source\repos\MLDersleri\MulticlassClassificationIris\Data";//Veri setlerinin temel yolunu belirler.
string TrainDataRelativePath = $"{BaseDatasetsRelativePath}/iris-train.txt";//Eğitim veri setinin yolunu belirler.
string TestDataRelativePath = $"{BaseDatasetsRelativePath}/iris-test.txt";//Test veri setinin yolunu belirler.
string TrainDataPath = GetAbsolutePath(TrainDataRelativePath);//Eğitim veri setinin mutlak yolunu alır.
string TestDataPath = GetAbsolutePath(TestDataRelativePath);//Test veri setinin mutlak yolunu alır.
string BaseModelsRelativePath = @"C:\Users\firat\source\repos\MLDersleri\MulticlassClassificationIris\MLModels";//Modelin kaydedileceği dizini belirler.
string ModelRelativePath = $"{BaseModelsRelativePath}/IrisClassificationModel.zip";//Modelin kaydedileceği dosya adını belirler.
string ModelPath = GetAbsolutePath(ModelRelativePath);//Modelin mutlak yolunu alır.

MLContext mlContext = new MLContext(seed: 0);
//ML.NET bağlamını oluşturur, makine öğrenimi işlemlerinin başlangıç noktasıdır.
BuildTrainEvaluateAndSaveModel(mlContext);
//Modeli oluşturur, eğitir, değerlendirir ve kaydeder.
TestSomePredictions(mlContext);
//Kaydedilen modelle bazı tahminler yapar ve sonuçları ekrana yazdırır.
Console.WriteLine("=============== End of process, hit any key to finish ===============");
Console.ReadKey();
//Input almayı bekletir.
void BuildTrainEvaluateAndSaveModel(MLContext mlContext)
{
    IDataView trainingDataView = mlContext.Data.LoadFromTextFile<IrisData>(TrainDataPath, hasHeader: true);
    //Eğitim verilerini yükler.

    IDataView testDataView = mlContext.Data.LoadFromTextFile<IrisData>(TestDataPath, hasHeader: true);
    //Test verilerini yükler.

    EstimatorChain<TransformerChain<ColumnConcatenatingTransformer>> dataProcessPipeline = mlContext.Transforms.Conversion
        .MapValueToKey(outputColumnName: "KeyColumn", inputColumnName: nameof(IrisData.Label))
        .Append(mlContext.Transforms
        .Concatenate("Features", nameof(IrisData.SepalLength), nameof(IrisData.SepalWidth), nameof(IrisData.PetalLength), nameof(IrisData.PetalWidth))
        .AppendCacheCheckpoint(mlContext));
    //Etiketleri anahtara dönüştürür ve özellikleri birleştirir.

    EstimatorChain<KeyToValueMappingTransformer> trainer = mlContext.MulticlassClassification.Trainers
        .SdcaMaximumEntropy(labelColumnName: "KeyColumn", featureColumnName: "Features")
        .Append(mlContext.Transforms.Conversion
        .MapKeyToValue(outputColumnName: nameof(IrisData.Label), inputColumnName: "KeyColumn"));
    //SDCA algoritması ile sınıflandırma modeli oluşturur ve etiketleri geri dönüştürür.

    EstimatorChain<TransformerChain<KeyToValueMappingTransformer>> trainingPipeline = dataProcessPipeline.Append(trainer);
    Console.WriteLine("=============== Training the model ===============");
    ITransformer trainedModel = trainingPipeline.Fit(trainingDataView);
    Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
    IDataView predictions = trainedModel.Transform(testDataView);
    MulticlassClassificationMetrics metrics = mlContext.MulticlassClassification.Evaluate(predictions, "Label", "Score");
    ConsoleHelper.PrintMultiClassClassificationMetrics(trainer.ToString(), metrics);
    //Modeli eğitir, tahminler yapar ve modelin performansını değerlendirir.

    mlContext.Model.Save(trainedModel, trainingDataView.Schema, ModelPath);
    Console.WriteLine("The model is saved to {0}", ModelPath);
    //Eğitilmiş modeli belirtilen yola kaydeder.
}

void TestSomePredictions(MLContext mlContext)
{
    ITransformer trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);
    //Kaydedilen modeli yükler.

    PredictionEngine<IrisData, IrisPrediction> predEngine = mlContext.Model.CreatePredictionEngine<IrisData, IrisPrediction>(trainedModel);
    //Tahmin motoru oluşturur.

    VBuffer<float> keys = default;
    predEngine.OutputSchema["PredictedLabel"].GetKeyValues(ref keys);
    float[] labelsArray = keys.DenseValues().ToArray();
    //Modelin etiket anahtarlarını alır.

    Dictionary<float, string> IrisFlowers = new Dictionary<float, string>();
    IrisFlowers.Add(0, "Setosa");
    IrisFlowers.Add(1, "versicolor");
    IrisFlowers.Add(2, "virginica");
    //Anahtar değerlerine karşılık gelen iris çiçek türlerini tanımlar.

    Console.WriteLine("=====Predicting using model====");

    IrisPrediction resultprediction1 = predEngine.Predict(SampleIrisData.Iris1);
    //SampleIrisData nesneleri üzerinde tahminler yapar.

    Console.WriteLine($"Actual: setosa.     Predicted label and score:  {IrisFlowers[labelsArray[0]]}: {resultprediction1.Score[0]:0.####}");
    Console.WriteLine($"                                                {IrisFlowers[labelsArray[1]]}: {resultprediction1.Score[1]:0.####}");
    Console.WriteLine($"                                                {IrisFlowers[labelsArray[2]]}: {resultprediction1.Score[2]:0.####}");
    Console.WriteLine();
    //Sonuçları ekrana yazdırır.

    IrisPrediction resultprediction2 = predEngine.Predict(SampleIrisData.Iris2);

    Console.WriteLine($"Actual: Virginica.   Predicted label and score:  {IrisFlowers[labelsArray[0]]}: {resultprediction2.Score[0]:0.####}");
    Console.WriteLine($"                                                 {IrisFlowers[labelsArray[1]]}: {resultprediction2.Score[1]:0.####}");
    Console.WriteLine($"                                                 {IrisFlowers[labelsArray[2]]}: {resultprediction2.Score[2]:0.####}");
    Console.WriteLine();

    IrisPrediction resultprediction3 = predEngine.Predict(SampleIrisData.Iris3);

    Console.WriteLine($"Actual: Versicolor.   Predicted label and score: {IrisFlowers[labelsArray[0]]}: {resultprediction3.Score[0]:0.####}");
    Console.WriteLine($"                                                 {IrisFlowers[labelsArray[1]]}: {resultprediction3.Score[1]:0.####}");
    Console.WriteLine($"                                                 {IrisFlowers[labelsArray[2]]}: {resultprediction3.Score[2]:0.####}");
    Console.WriteLine();
    //SampleIrisData Class'ında Iris1, Iris2 ve Iris3 adında 3 tane çiçek datası var. 3'ü için de tahmin yaptırıp sonuçları konsola yazdırdık.
}

string GetAbsolutePath(string relativePath)//Verilen göreceli yolu, programın çalıştığı dizine göre mutlak yola dönüştürür.
{
    FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
    string assemblyFolderPath = _dataRoot.Directory.FullName;
    string fullPath = Path.Combine(assemblyFolderPath, relativePath);
    return fullPath;
}
