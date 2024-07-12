using Common;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML;
using Microsoft.ML.Trainers;
using CustomerSegmentation.Train.DataStructures;

//Dosya Yollarının Tanımlanması;
string assetsRelativePath = @"C:\Users\firat\source\repos\MLDersleri\CustomerSegmentation.Train\Assets";
string assetsPath = GetAbsolutePath(assetsRelativePath);
string transactionsCsv = Path.Combine(assetsPath, "Inputs", "transactions.csv");
string offersCsv = Path.Combine(assetsPath, "Inputs", "offers.csv");
string pivotCsv = Path.Combine(assetsPath, "Inputs", "pivot.csv");
string modelPath = Path.Combine(assetsPath, "Outputs", "retailClustering.zip");

try
{
    //Bu satır, offersCsv ve transactionsCsv dosyalarından gelen verileri ön işleyip pivotCsv dosyasına kaydeder;
    DataHelpers.PreProcessAndSave(offersCsv, transactionsCsv, pivotCsv);

    //MLContext: ML.NET bağlamını temsil eder ve çeşitli işlemler için kullanılır;
    MLContext mlContext = new MLContext(seed: 1);

    //pivotCsv dosyasından verileri yükler. Bu işlem sırasında Features ve LastName sütunları tanımlanır.
    IDataView pivotDataView = mlContext.Data.LoadFromTextFile(path: pivotCsv, columns: new[]
    {
        new TextLoader.Column("Features", DataKind.Single, new[] {new TextLoader.Range(0, 31) }),
        new TextLoader.Column(nameof(PivotData.LastName), DataKind.String, 32)
    }, hasHeader: true, separatorChar: ',');

    //Bu kısım, veri işleme pipeline'ını tanımlar.
    EstimatorChain<OneHotEncodingTransformer> dataProcessPipeline = mlContext.Transforms
        .ProjectToPrincipalComponents(outputColumnName: "PCAFeatures", inputColumnName: "Features", rank: 2)//PCA ile özellikleri iki boyuta indirger.
        .Append(mlContext.Transforms.Categorical
        .OneHotEncoding(outputColumnName: "LastNameKey", inputColumnName: nameof(PivotData.LastName), OneHotEncodingEstimator.OutputKind.Indicator));//LastName sütununu one-hot encoding ile dönüştürür.

    //PeekDataViewInConsole ve PeekVectorColumnDataInConsole: Verileri konsolda incelemek için kullanılır.
    ConsoleHelper.PeekDataViewInConsole(mlContext, pivotDataView, dataProcessPipeline, 10);
    ConsoleHelper.PeekVectorColumnDataInConsole(mlContext, "Features", pivotDataView, dataProcessPipeline, 10);

    //KMeans algoritması ile kümeleme modelini tanımlar;
    KMeansTrainer trainer = mlContext.Clustering.Trainers.KMeans(featureColumnName: "Features", numberOfClusters: 3);

    //Veri işleme pipeline'ına eğitim algoritmasını ekler;
    EstimatorChain<ClusteringPredictionTransformer<KMeansModelParameters>> trainingPipeline = dataProcessPipeline.Append(trainer);

    //Modelin Eğitilmesi;
    Console.WriteLine("=============== Training the model ===============");
    ITransformer trainedModel = trainingPipeline.Fit(pivotDataView);//Fit: Modeli eğitir.

    Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
    IDataView predictions = trainedModel.Transform(pivotDataView);//Transform: Eğitilmiş modeli kullanarak tahminler yapar.
    ClusteringMetrics metrics = mlContext.Clustering.Evaluate(predictions, scoreColumnName: "Score", featureColumnName: "Features");//Evaluate: Modelin performansını değerlendirir.

    //ConsoleHelper metodu;
    ConsoleHelper.PrintClusteringMetrics(trainer.ToString(), metrics);

    //Modelin Kaydedilmesi;
    mlContext.Model.Save(trainedModel, pivotDataView.Schema, modelPath);//Save: Eğitilmiş modeli belirtilen dosya yoluna kaydeder.

    Console.WriteLine("The model is saved to {0}", modelPath);
}
catch (Exception ex)//Hataları yakalar ve konsolda gösterir.
{
    ConsoleHelper.ConsoleWriteException(ex.ToString());
}

ConsoleHelper.ConsolePressAnyKey();//Programın sonlanmadan önce kullanıcıdan bir tuşa basmasını bekler.

string GetAbsolutePath(string relativePath)//Verilen göreli dosya yolunu mutlak dosya yoluna çevirir.
{
    FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
    string assemblyFolderPath = _dataRoot.Directory.FullName;
    string fullPath = Path.Combine(assemblyFolderPath, relativePath);
    return fullPath;
}
