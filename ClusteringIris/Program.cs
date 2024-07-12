using ClusteringIris.DataStructures;
using Common;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

//Path'ler;
string AppPath = Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
string BaseDatasetsRelativePath = @"C:\Users\firat\source\repos\MLDersleri\ClusteringIris\Data";
string DataSetRealtivePath = $"{BaseDatasetsRelativePath}/iris-full.txt";
string DataPath = GetAbsolutePath(DataSetRealtivePath);
string BaseModelsRelativePath = @"C:\Users\firat\source\repos\MLDersleri\ClusteringIris\MLModels";
string ModelRelativePath = $"{BaseModelsRelativePath}/IrisModel.zip";
string ModelPath = GetAbsolutePath(ModelRelativePath);

IDataView trainingDataView;//Eğitim veri kümesini tutacak değişkeni tanımlar.
IDataView testingDataView;//Test veri kümesini tutacak değişkeni tanımlar.

MLContext mlContext = new MLContext(seed: 1);//ML.NET bağlamını oluşturur ve rastgele işlemler için tohum değeri (seed) belirler.

//Veri setini metin dosyasından yükler. Dosyanın yolu DataPath'ten alınır, sütunlar ve veri türleri belirtilir, ilk satırın başlık olduğu belirtilir ve sütunların tab karakteriyle ayrıldığı belirtilir.
IDataView fullData = mlContext.Data.LoadFromTextFile(path: DataPath, columns: new[]
{
    new TextLoader.Column("Label", DataKind.Single, 0),
    new TextLoader.Column(nameof(IrisData.SepalLength), DataKind.Single, 1),
    new TextLoader.Column(nameof(IrisData.SepalWidth), DataKind.Single, 2),
    new TextLoader.Column(nameof(IrisData.PetalLength), DataKind.Single, 3),
    new TextLoader.Column(nameof(IrisData.PetalWidth), DataKind.Single, 4),
}, hasHeader: true, separatorChar: '\t');

DataOperationsCatalog.TrainTestData trainTestData = mlContext.Data.TrainTestSplit(fullData, testFraction: 0.2);//Veri setini eğitim ve test olarak %80-%20 oranında böler;
trainingDataView = trainTestData.TrainSet;//Eğitim veri kümesini trainingDataView değişkenine atar.
testingDataView = trainTestData.TestSet;//Test veri kümesini testingDataView değişkenine atar.

//Sepal ve petal özelliklerini birleştirerek "Features" adlı tek bir özellik vektörü oluşturur;
ColumnConcatenatingEstimator dataProcessPipeline =
    mlContext.Transforms.Concatenate("Features", nameof(IrisData.SepalLength), nameof(IrisData.SepalWidth), nameof(IrisData.PetalLength), nameof(IrisData.PetalWidth));

//Eğitim veri kümesinin ilk 10 satırını konsola yazdırır;
ConsoleHelper.PeekDataViewInConsole(mlContext, trainingDataView, dataProcessPipeline, 10);
//"Features" sütununun ilk 10 satırını konsola yazdırır;
ConsoleHelper.PeekVectorColumnDataInConsole(mlContext, "Features", trainingDataView, dataProcessPipeline, 10);

//"Features", numberOfClusters: 3);: K-Means algoritması ile 3 küme oluşturacak bir eğitici (trainer) tanımlar;
KMeansTrainer trainer = mlContext.Clustering.Trainers.KMeans(featureColumnName: "Features", numberOfClusters: 3);
//Veri işleme ve eğitim aşamalarını içeren bir boru hattı (pipeline) oluşturur;
EstimatorChain<ClusteringPredictionTransformer<KMeansModelParameters>> trainingPipeline = dataProcessPipeline.Append(trainer);
//Eğitim veri kümesi üzerinde boru hattını çalıştırarak modeli eğitir;
TransformerChain<ClusteringPredictionTransformer<KMeansModelParameters>> trainedModel = trainingPipeline.Fit(trainingDataView);

//Eğitimli modeli test veri kümesine uygular ve tahminleri alır;
IDataView predictions = trainedModel.Transform(testingDataView);
//Test veri kümesi üzerinde modelin performansını değerlendirir;
ClusteringMetrics metrics = mlContext.Clustering.Evaluate(predictions, scoreColumnName: "Score", featureColumnName: "Features");
//Modelin değerlendirme metriklerini konsola yazdırır;
ConsoleHelper.PrintClusteringMetrics(trainer.ToString(), metrics);
//Eğitilmiş modeli belirtilen yola kaydeder;
mlContext.Model.Save(trainedModel, trainingDataView.Schema, ModelPath);
//Eğitim sürecinin bittiğini ve tek bir örnek için tahmin yapılacağını belirtir;
Console.WriteLine("=============== End of training process ===============");
Console.WriteLine("=============== Predict a cluster for a single case (Single Iris data sample) ===============");

//Tahmin yapmak için kullanılacak bir iris çiçeği örneği oluşturur;
IrisData sampleIrisData = new IrisData()
{
    SepalLength = 3.3f,
    SepalWidth = 1.6f,
    PetalLength = 0.2f,
    PetalWidth = 5.1f,
};

//Kaydedilmiş modeli belirtilen yoldan yükler ve modelin giriş şemasını (schema) alır.
ITransformer model = mlContext.Model.Load(ModelPath, out DataViewSchema modelInputSchema);

//Yüklenmiş modelden tek bir örnek için tahmin yapmak amacıyla bir PredictionEngine oluşturur;
PredictionEngine<IrisData, IrisPrediction> predEngine = mlContext.Model.CreatePredictionEngine<IrisData, IrisPrediction>(model);

//sampleIrisData örneği için modelden tahmin yapar ve sonucu resultprediction değişkenine atar;
IrisPrediction resultprediction = predEngine.Predict(sampleIrisData);

//Konsol Metotları;
Console.WriteLine($"Cluster assigned for setosa flowers:" + resultprediction.SelectedClusterId);//Tahmin edilen küme ID'sini konsola yazdırır.
Console.WriteLine("=============== End of process, hit any key to finish ===============");//İşlemin bittiğini belirtir.
Console.ReadKey();//Kullanıcıdan bir tuşa basmasını bekler, böylece konsol penceresi hemen kapanmaz.

//GetAbsolutePath: İlgili dosya yollarını mutlak yola dönüştüren bir yardımcı metot. Uygulamanın çalıştırıldığı dizine göre verilen göreli yolu mutlak yola çevirir;
string GetAbsolutePath(string relativePath)
{
    FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
    string assemblyFolderPath = _dataRoot.Directory.FullName;
    string fullPath = Path.Combine(assemblyFolderPath, relativePath);
    return fullPath;
}
