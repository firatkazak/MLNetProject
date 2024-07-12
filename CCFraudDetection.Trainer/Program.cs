using Common;
using CCFraudDetection.Common.DataModels;
using System.IO.Compression;
using Microsoft.ML;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Data;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Transforms;
using static Microsoft.ML.DataOperationsCatalog;

string AssetsRelativePath = @"C:\Users\firat\source\repos\MLDersleri\CCFraudDetection.Trainer\Assets";
//Projenin varlıklarının bulunduğu relatif yol.
string assetsPath = GetAbsolutePath(AssetsRelativePath);
//Tam yolun hesaplanması.
string zipDataSet = Path.Combine(assetsPath, "Input", "creditcardfraud-dataset.zip");
//ZIP dosyasının yolu.
string fullDataSetFilePath = Path.Combine(assetsPath, "Input", "creditcard.csv");
//Veri setinin tam dosya yolu.
string trainDataSetFilePath = Path.Combine(assetsPath, "Output", "trainData.csv");
//Eğitim veri setinin yolu.
string testDataSetFilePath = Path.Combine(assetsPath, "Output", "testData.csv");
//Test veri setinin yolu.
string modelFilePath = Path.Combine(assetsPath, "Output", "fastTree.zip");
//Eğitilen modelin kaydedileceği dosya yolu.

UnZipDataSet(zipDataSet, fullDataSetFilePath);
//ZIP dosyasını belirtilen hedefe çıkarır.
MLContext mlContext = new MLContext(seed: 1);
//ML.NET bağlamını başlatır. Tohum değeri (seed) rastgele işlemler için tekrar edilebilirlik sağlar.
PrepDatasets(mlContext, fullDataSetFilePath, trainDataSetFilePath, testDataSetFilePath);
//Eğitim ve test veri setlerini hazırlar.
IDataView trainingDataView = mlContext.Data.LoadFromTextFile<TransactionObservation>(trainDataSetFilePath, separatorChar: ',', hasHeader: true);
//CSV dosyalarını yükler ve IDataView döner.
IDataView testDataView = mlContext.Data.LoadFromTextFile<TransactionObservation>(testDataSetFilePath, separatorChar: ',', hasHeader: true);
//CSV dosyalarını yükler ve IDataView döner.
(ITransformer model, string trainerName) = TrainModel(mlContext, trainingDataView);
//Modeli eğitir ve eğitilen modeli döner.
EvaluateModel(mlContext, model, testDataView, trainerName);
//Modelin doğruluğunu test verisiyle değerlendirir.
SaveModel(mlContext, model, modelFilePath, trainingDataView.Schema);
//Eğitilen modeli belirtilen dosya yoluna kaydeder.
Console.WriteLine("=============== Press any key ===============");
//Press any key yazar.
Console.ReadKey();
//Kullanıcıdan giriş bekler.
void PrepDatasets(MLContext mlContext, string fullDataSetFilePath, string trainDataSetFilePath, string testDataSetFilePath)
{
    if (!File.Exists(trainDataSetFilePath) && !File.Exists(testDataSetFilePath))
    {
        Console.WriteLine("===== Preparing train/test datasets =====");

        IDataView originalFullData = mlContext.Data.LoadFromTextFile<TransactionObservation>(fullDataSetFilePath, separatorChar: ',', hasHeader: true);
        //Input klasöründeki creditcard.csv dosyasını alıyor.
        TrainTestData trainTestData = mlContext.Data.TrainTestSplit(originalFullData, testFraction: 0.2, seed: 1);
        //Veri setini eğitim ve test olarak böler.
        IDataView trainData = trainTestData.TrainSet;
        //trainTestData değişkeni, eğitim ve test veri setlerine ayrılmış veriyi temsil eder.
        //TrainSet özelliği, bu veri setinin eğitim (train) kısmını içerir. Yani bu satır, eğitim verilerini trainData değişkenine atar.
        IDataView testData = trainTestData.TestSet;
        //trainTestData değişkeninin test kısmını testData değişkenine atar.
        //Test verileri, modelin performansını değerlendirmek için kullanılır.
        InspectData(mlContext, testData, 4);
        //testData değişkenini incelemek için InspectData adlı fonksiyonu çağırır. Bu fonksiyon, testData veri setinden 4 kayıt alarak içeriğini konsolda gösterir. Amaç, veri setinin doğru şekilde yüklendiğini ve beklenen formatta olduğunu kontrol etmektir.
        using (FileStream fileStream = File.Create(trainDataSetFilePath))//trainDataSetFilePath dosyasını oluşturur ve bir FileStream nesnesi döner.
        {
            mlContext.Data.SaveAsText(trainData, fileStream, separatorChar: ',', headerRow: true, schema: true);
            //trainData veri setini verilen fileStream üzerinden CSV formatında yazar.
            //separatorChar: ',' ile CSV dosyasında verilerin virgülle ayrılacağını belirler.
            //headerRow: true ile CSV dosyasına sütun başlıklarının yazılacağını ve schema: true ile veri şemasının (veri türleri) dosyaya dahil edileceğini belirtir.
        }//Bu blok, trainData veri setini bir CSV dosyasına kaydeder.

        using (FileStream fileStream = File.Create(testDataSetFilePath))
        {
            mlContext.Data.SaveAsText(testData, fileStream, separatorChar: ',', headerRow: true, schema: true);
        }//Bu blok, yukarıdaki blokla aynıdır ancak testData veri setini testDataSetFilePath dosyasına kaydeder. Yani test verilerini bir CSV dosyasına yazar. Aynı şekilde, dosya oluşturulur, testData veri seti CSV formatında dosyaya yazılır, virgül ayırıcı olarak kullanılır, sütun başlıkları ve veri şeması dosyaya dahil edilir.
    }
}

//Append(): Veri işleme ve model eğitimi pipeline'larını birleştirmek için kullanılır. Bir işlem hattı, veri dönüşümlerini ve model eğitim adımlarını zincirleme şekilde uygulayarak bir veri setine işlem yapar. Append metodu, bu adımları ardışık olarak eklemenizi sağlar.

//TrainModel(): Modeli Eğittiğimiz Metot.
(ITransformer model, string trainerName) TrainModel(MLContext mlContext, IDataView trainDataView)
{
    string[] featureColumnNames = trainDataView.Schema
        .AsQueryable()
        .Select(column => column.Name)
        .Where(name => name != nameof(TransactionObservation.Label))
        .Where(name => name != "IdPreservationColumn")
        .Where(name => name != "Time")
        .ToArray();

    IEstimator<ITransformer> dataProcessPipeline = mlContext.Transforms
        .Concatenate("Features", featureColumnNames)//Özellik sütunlarını birleştirir.
        .Append(mlContext.Transforms.DropColumns(new string[] { "Time" }))//Zaman sütununu düşürür.
        .Append(mlContext.Transforms.NormalizeMeanVariance(inputColumnName: "Features", outputColumnName: "FeaturesNormalizedByMeanVar"));
    //NormalizeMeanVariance: Özellikleri normalize eder.
    ConsoleHelper.PeekDataViewInConsole(mlContext, trainDataView, dataProcessPipeline, 2);
    //
    ConsoleHelper.PeekVectorColumnDataInConsole(mlContext, "Features", trainDataView, dataProcessPipeline, 1);
    //
    FastTreeBinaryTrainer trainer = mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: nameof(TransactionObservation.Label), featureColumnName: "FeaturesNormalizedByMeanVar", numberOfLeaves: 20, numberOfTrees: 100, minimumExampleCountPerLeaf: 10, learningRate: 0.2);
    //FastTreeBinaryTrainer: Modeli eğitir.
    EstimatorChain<BinaryPredictionTransformer<CalibratedModelParametersBase<FastTreeBinaryModelParameters, PlattCalibrator>>> trainingPipeline
        = dataProcessPipeline.Append(trainer);
    //
    ConsoleHelper.ConsoleWriteHeader("=============== Training model ===============");

    TransformerChain<BinaryPredictionTransformer<CalibratedModelParametersBase<FastTreeBinaryModelParameters, PlattCalibrator>>> model = trainingPipeline.Fit(trainDataView);

    ConsoleHelper.ConsoleWriteHeader("=============== End of training process ===============");

    TransformerChain<FeatureContributionCalculatingTransformer> fccPipeline = model
        .Append(mlContext.Transforms
        .CalculateFeatureContribution(model.LastTransformer)
        .Fit(dataProcessPipeline.Fit(trainDataView).Transform(trainDataView)));
    //Verilen veri ile eğitir ve modeli döner.
    return (fccPipeline, fccPipeline.ToString());
}

//EvaluateModel(): Modeli Değerlendiren metot.
void EvaluateModel(MLContext mlContext, ITransformer model, IDataView testDataView, string trainerName)
{
    Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");

    IDataView predictions = model.Transform(testDataView);
    //Transform(): Test verisini kullanarak tahminler yapar.
    CalibratedBinaryClassificationMetrics metrics =
        mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: nameof(TransactionObservation.Label), scoreColumnName: "Score");
    //Evaluate(): Modelin performansını değerlendirir ve metrikleri döner.
    ConsoleHelper.PrintBinaryClassificationMetrics(trainerName, metrics);
}

//GetAbsolutePath(): Veri yolunu çeken metot.
string GetAbsolutePath(string relativePath)
{
    FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);

    string assemblyFolderPath = _dataRoot.Directory.FullName;

    string fullPath = Path.Combine(assemblyFolderPath, relativePath);

    return fullPath;
}

//InspectData(): Verileri Denetleyecek olan metot.
void InspectData(MLContext mlContext, IDataView data, int records)
{
    Console.WriteLine("Show 4 fraud transactions (true)");

    ShowObservationsFilteredByLabel(mlContext, data, label: true, count: records);

    Console.WriteLine("Show 4 NOT-fraud transactions (false)");

    ShowObservationsFilteredByLabel(mlContext, data, label: false, count: records);
}

//ShowObservationsFilteredByLabel(): Belirtilen etiket ile verileri filtreler ve belirtilen sayıda işlemi gösterir.
void ShowObservationsFilteredByLabel(MLContext mlContext, IDataView dataView, bool label = true, int count = 2)
{
    List<TransactionObservation> data =
        mlContext.Data.CreateEnumerable<TransactionObservation>(dataView, reuseRowObject: false).Where(x => x.Label == label).Take(count).ToList();
    //CreateEnumerable: IDataView verisini enumerable olarak döner.
    //Where: Verileri etiketle filtreler.
    //Take: Belirtilen sayıda veri alır.
    data.ForEach(row => { row.PrintToConsole(); });
    //PrintToConsole: Veriyi konsola yazdırır.
}

//UnZipDataSet(): Zip dosyasını çıkaracak metot.
void UnZipDataSet(string zipDataSet, string destinationFile)
{
    if (!File.Exists(destinationFile))
    {
        string destinationDirectory = Path.GetDirectoryName(destinationFile);
        //çıkartılacak dizinin yolu.
        ZipFile.ExtractToDirectory(zipDataSet, $"{destinationDirectory}");
        //ExtractToDirectory(): ZIP dosyasını belirtilen dizine çıkarır.
    }
}

//SaveModel(): Eğitilen modeli kaydedecek metot.
void SaveModel(MLContext mlContext, ITransformer model, string modelFilePath, DataViewSchema trainingDataSchema)
{
    mlContext.Model.Save(model, trainingDataSchema, modelFilePath);
    ////Save(): Eğitilen modeli belirtilen dosya yoluna kaydeder.
    Console.WriteLine("Saved model to " + modelFilePath);
}
