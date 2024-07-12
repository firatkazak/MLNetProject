using ICSharpCode.SharpZipLib.GZip;
using ICSharpCode.SharpZipLib.Tar;
using LargeDatasets.DataStructures;
using Microsoft.ML;
using System.Net;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using Common;

string originalDataDirectoryRelativePath = @"../../../Data/OriginalUrlData";
string originalDataReltivePath = @"../../../Data/OriginalUrlData/url_svmlight";
string preparedDataReltivePath = @"../../../Data/PreparedUrlData/url_svmlight";
string originalDataDirectoryPath = GetAbsolutePath(originalDataDirectoryRelativePath);
string originalDataPath = GetAbsolutePath(originalDataReltivePath);
string preparedDataPath = GetAbsolutePath(preparedDataReltivePath);

DownloadDataset(originalDataDirectoryPath);

PrepareDataset(originalDataPath, preparedDataPath);

MLContext mlContext = new MLContext();

IDataView fullDataView = mlContext.Data.LoadFromTextFile<UrlData>(path: Path.Combine(preparedDataPath, "*"), hasHeader: false, allowSparse: true);

TrainTestData trainTestData = mlContext.Data.TrainTestSplit(fullDataView, testFraction: 0.2, seed: 1);
IDataView trainDataView = trainTestData.TrainSet;
IDataView testDataView = trainTestData.TestSet;

Dictionary<string, bool> UrlLabelMap = new Dictionary<string, bool>();
UrlLabelMap["+1"] = true;
UrlLabelMap["-1"] = false;
ValueMappingEstimator<string, bool> dataProcessingPipeLine = mlContext.Transforms.Conversion.MapValue("LabelKey", UrlLabelMap, "LabelColumn");
ConsoleHelper.PeekDataViewInConsole(mlContext, trainDataView, dataProcessingPipeLine, 2);

EstimatorChain<FieldAwareFactorizationMachinePredictionTransformer> trainingPipeLine =
    dataProcessingPipeLine.Append(mlContext.BinaryClassification.Trainers.FieldAwareFactorizationMachine(labelColumnName: "LabelKey", featureColumnName: "FeatureVector"));

Console.WriteLine("====Training the model=====");
TransformerChain<FieldAwareFactorizationMachinePredictionTransformer> mlModel = trainingPipeLine.Fit(trainDataView);
Console.WriteLine("====Completed Training the model=====");
Console.WriteLine("");

Console.WriteLine("====Evaluating the model=====");
IDataView predictions = mlModel.Transform(testDataView);
CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "LabelKey", scoreColumnName: "Score");
ConsoleHelper.PrintBinaryClassificationMetrics(mlModel.ToString(), metrics);

Console.WriteLine("====Predicting sample data=====");
PredictionEngine<UrlData, UrlPrediction> predEngine = mlContext.Model.CreatePredictionEngine<UrlData, UrlPrediction>(mlModel);
List<UrlData> sampleDatas = CreateSingleDataSample(mlContext, trainDataView);

foreach (UrlData sampleData in sampleDatas)
{
    UrlPrediction predictionResult = predEngine.Predict(sampleData);
    Console.WriteLine($"Single Prediction --> Actual value: {sampleData.LabelColumn} | Predicted value: {predictionResult.Prediction}");
}

Console.WriteLine("====End of Process..Press any key to exit====");
Console.ReadLine();

void DownloadDataset(string originalDataDirectoryPath)
{
    if (!Directory.Exists(originalDataDirectoryPath))
    {
        Console.WriteLine("====Downloading and extracting data====");
        using (WebClient client = new WebClient())
        {
            client.DownloadFile("https://archive.ics.uci.edu/ml/machine-learning-databases/url/url_svmlight.tar.gz", "url_svmlight.zip");
        }

        Stream inputStream = File.OpenRead("url_svmlight.zip");
        Stream gzipStream = new GZipInputStream(inputStream);
        TarArchive tarArchive = TarArchive.CreateInputTarArchive(gzipStream);
        tarArchive.ExtractContents(originalDataDirectoryPath);

        tarArchive.Close();
        gzipStream.Close();
        inputStream.Close();
        Console.WriteLine("====Downloading and extracting is completed====");
    }
}

void PrepareDataset(string originalDataPath, string preparedDataPath)
{
    if (!Directory.Exists(preparedDataPath))
    {
        Directory.CreateDirectory(preparedDataPath);
    }

    Console.WriteLine("====Preparing Data====");
    Console.WriteLine("");

    if (Directory.GetFiles(preparedDataPath).Length == 0)
    {
        List<string> ext = new List<string> { ".svm" };
        IEnumerable<string> filesInDirectory = Directory.GetFiles(originalDataPath, "*.*", SearchOption.AllDirectories).Where(s => ext.Contains(Path.GetExtension(s)));
        
        foreach (string file in filesInDirectory)
        {
            AddFeaturesColumn(Path.GetFullPath(file), preparedDataPath);
        }
    }
    Console.WriteLine("====Data Preparation is done====");
    Console.WriteLine("");
    Console.WriteLine("original data path= {0}", originalDataPath);
    Console.WriteLine("");
    Console.WriteLine("prepared data path= {0}", preparedDataPath);
    Console.WriteLine("");
}

void AddFeaturesColumn(string sourceFilePath, string preparedDataPath)
{
    string sourceFileName = Path.GetFileName(sourceFilePath);
    string preparedFilePath = Path.Combine(preparedDataPath, sourceFileName);

    if (!File.Exists(preparedFilePath))
    {
        File.Copy(sourceFilePath, preparedFilePath, true);
    }

    string newColumnData = "3231961";
    string[] CSVDump = File.ReadAllLines(preparedFilePath);
    List<List<string>> CSV = CSVDump.Select(x => x.Split(' ').ToList()).ToList();

    for (int i = 0; i < CSV.Count; i++)
    {
        CSV[i].Insert(1, newColumnData);
    }
    File.WriteAllLines(preparedFilePath, CSV.Select(x => string.Join('\t', x)));
}

string GetAbsolutePath(string relativePath)
{
    FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
    string assemblyFolderPath = _dataRoot.Directory.FullName;
    string fullPath = Path.Combine(assemblyFolderPath, relativePath);
    return fullPath;
}
List<UrlData> CreateSingleDataSample(MLContext mlContext, IDataView dataView)
{
    List<UrlData> sampleForPredictions = mlContext.Data.CreateEnumerable<UrlData>(dataView, false).Take(4).ToList(); ;
    return sampleForPredictions;
}
