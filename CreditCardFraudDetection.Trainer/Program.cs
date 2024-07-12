using Common;
using CreditCardFraudDetection.Common.DataModels;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System.IO.Compression;
using static Microsoft.ML.DataOperationsCatalog;

//Yollar;
string AssetsRelativePath = @"C:\Users\firat\source\repos\MLDersleri\CreditCardFraudDetection.Trainer\assets";
string assetsPath = GetAbsolutePath(AssetsRelativePath);
string zipDataSet = Path.Combine(assetsPath, "input", "creditcardfraud-dataset.zip");
string fullDataSetFilePath = Path.Combine(assetsPath, "input", "creditcard.csv");
string trainDataSetFilePath = Path.Combine(assetsPath, "output", "trainData.csv");
string testDataSetFilePath = Path.Combine(assetsPath, "output", "testData.csv");
string modelFilePath = Path.Combine(assetsPath, "output", "randomizedPca.zip");

//Sıkıştırılmış veri setini açar;
UnZipDataSet(zipDataSet, fullDataSetFilePath);

//Yeni bir ML.NET bağlamı oluşturur.
MLContext mlContext = new MLContext(seed: 1);

//Veri setlerini hazırlar.
PrepDatasets(mlContext, fullDataSetFilePath, trainDataSetFilePath, testDataSetFilePath);

//Eğitim veri setini yükler;
IDataView trainingDataView = mlContext.Data.LoadFromTextFile<TransactionObservation>(trainDataSetFilePath, separatorChar: ',', hasHeader: true);

//Test veri setini yükler;
IDataView testDataView = mlContext.Data.LoadFromTextFile<TransactionObservation>(testDataSetFilePath, separatorChar: ',', hasHeader: true);

//Modeli eğitir;
ITransformer model = TrainModel(mlContext, trainingDataView);

//Modeli değerlendirir;
EvaluateModel(mlContext, model, testDataView);

//Modeli kaydeder;
SaveModel(mlContext, model, modelFilePath, trainingDataView.Schema);

//Kullanıcıdan bir tuşa basmasını bekler;
Console.WriteLine("=============== Press any key ===============");
Console.ReadKey();

//Veri setlerini hazırlar;
void PrepDatasets(MLContext mlContext, string fullDataSetFilePath, string trainDataSetFilePath, string testDataSetFilePath)
{
    if (!File.Exists(trainDataSetFilePath) && !File.Exists(testDataSetFilePath))//Eğitim ve test veri seti dosyaları yoksa;
    {
        Console.WriteLine("===== Preparing train/test datasets =====");

        //Tam veri setini yükler;
        IDataView originalFullData = mlContext.Data.LoadFromTextFile<TransactionObservation>(fullDataSetFilePath, separatorChar: ',', hasHeader: true);

        //Veriyi eğitim ve test setlerine böler.
        TrainTestData trainTestData = mlContext.Data.TrainTestSplit(originalFullData, testFraction: 0.2, seed: 1);

        //Eğitim seti;
        IDataView trainData = trainTestData.TrainSet;

        //Test seti;
        IDataView testData = trainTestData.TestSet;

        //Test verilerini inceleyip konsola yazdırır;
        InspectData(mlContext, testData, 4);

        //Eğitim verilerini kaydeder;
        using (FileStream fileStream = File.Create(trainDataSetFilePath))
        {
            mlContext.Data.SaveAsText(trainData, fileStream, separatorChar: ',', headerRow: true, schema: true);
        }

        //Test verilerini kaydeder;
        using (FileStream fileStream = File.Create(testDataSetFilePath))
        {
            mlContext.Data.SaveAsText(testData, fileStream, separatorChar: ',', headerRow: true, schema: true);
        }
    }
}

//Modeli eğiten metot;
ITransformer TrainModel(MLContext mlContext, IDataView trainDataView)
{
    string[] featureColumnNames =//featureColumnNames: Özellik sütunlarının isimlerini alır;
        trainDataView.Schema
        .AsQueryable()
        .Select(column => column.Name)
        .Where(name => name != nameof(TransactionObservation.Label))
        .Where(name => name != "IdPreservationColumn")
        .Where(name => name != nameof(TransactionObservation.Time))
        .ToArray();

    IEstimator<ITransformer> dataProcessPipeline =//Veri işleme ardışık düzenini oluşturur;
        mlContext.Transforms
        .Concatenate("Features", featureColumnNames)
        .Append(mlContext.Transforms.DropColumns(new string[] { nameof(TransactionObservation.Time) }))
        .Append(mlContext.Transforms.NormalizeLpNorm(outputColumnName: "NormalizedFeatures", inputColumnName: "Features"));

    //Eğitim verilerini filtreler;
    IDataView normalTrainDataView = mlContext.Data.FilterRowsByColumn(trainDataView, columnName: nameof(TransactionObservation.Label), lowerBound: 0, upperBound: 1);

    //Verileri inceleyip konsola yazdırır;
    ConsoleHelper.PeekDataViewInConsole(mlContext, normalTrainDataView, dataProcessPipeline, 2);
    ConsoleHelper.PeekVectorColumnDataInConsole(mlContext, "NormalizedFeatures", normalTrainDataView, dataProcessPipeline, 2);

    //RandomizedPCA eğiticisinin seçeneklerini ayarlar;
    RandomizedPcaTrainer.Options options = new RandomizedPcaTrainer.Options
    {
        FeatureColumnName = "NormalizedFeatures",
        ExampleWeightColumnName = null,
        Rank = 28,
        Oversampling = 20,
        EnsureZeroMean = true,
        Seed = 1
    };

    //Eğiticiyi oluşturur;
    IEstimator<ITransformer> trainer = mlContext.AnomalyDetection.Trainers.RandomizedPca(options: options);

    //Eğitim ardışık düzenini oluşturur;
    EstimatorChain<ITransformer> trainingPipeline = dataProcessPipeline.Append(trainer);
    ConsoleHelper.ConsoleWriteHeader("=============== Training model ===============");

    //Modeli eğitir.
    TransformerChain<ITransformer> model = trainingPipeline.Fit(normalTrainDataView);
    ConsoleHelper.ConsoleWriteHeader("=============== End of training process ===============");

    // model'i döndürür.
    return model;
}

//Modeli değerlendiren metot;
void EvaluateModel(MLContext mlContext, ITransformer model, IDataView testDataView)
{
    Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");

    //Test verileri üzerinde tahminler yapar;
    IDataView predictions = model.Transform(testDataView);

    //Tahminleri değerlendirir;
    AnomalyDetectionMetrics metrics = mlContext.AnomalyDetection.Evaluate(predictions);

    //Değerlendirme sonuçlarını konsola yazdırır;
    ConsoleHelper.PrintAnomalyDetectionMetrics("RandomizedPca", metrics);
}

//Belirtilen sayıda sahtekarlık ve sahtekarlık olmayan işlemleri konsola yazdıran metot;
void InspectData(MLContext mlContext, IDataView data, int records)
{
    Console.WriteLine("Show 4 fraud transactions (true)");
    ShowObservationsFilteredByLabel(mlContext, data, label: true, count: records);
    Console.WriteLine("Show 4 NOT-fraud transactions (false)");
    ShowObservationsFilteredByLabel(mlContext, data, label: false, count: records);
}

//Belirli bir etikete göre filtrelenmiş işlemleri gösteren metot;
void ShowObservationsFilteredByLabel(MLContext mlContext, IDataView dataView, bool label = true, int count = 2)
{
    List<TransactionObservation> data = mlContext.Data
        .CreateEnumerable<TransactionObservation>(dataView, reuseRowObject: false)
        .Where(x => Math.Abs(x.Label - (label ? 1 : 0)) < float.Epsilon)
        .Take(count)
        .ToList();

    data.ForEach(row => { row.PrintToConsole(); });
}

//Sıkıştırılmış veri setini açan metot;
void UnZipDataSet(string zipDataSet, string destinationFile)
{
    if (!File.Exists(destinationFile))
    {
        string destinationDirectory = Path.GetDirectoryName(destinationFile);
        ZipFile.ExtractToDirectory(zipDataSet, $"{destinationDirectory}");
    }
}

//Modeli kaydeder;
void SaveModel(MLContext mlContext, ITransformer model, string modelFilePath, DataViewSchema trainingDataSchema)
{
    mlContext.Model.Save(model, trainingDataSchema, modelFilePath);
    Console.WriteLine("Saved model to " + modelFilePath);
}

//Göreceli yolu mutlak yola dönüştüren metot;
string GetAbsolutePath(string relativePath)
{
    FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
    string assemblyFolderPath = _dataRoot.Directory.FullName;
    string fullPath = Path.Combine(assemblyFolderPath, relativePath);
    return fullPath;
}
