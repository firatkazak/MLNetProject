using Common;
using GitHubLabeler;
using GitHubLabeler.DataStructures;
using Microsoft.Extensions.Configuration;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

string AppPath = Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);//Uygulamanın çalıştığı dizinin yolunu alır.
string BaseDatasetsRelativePath = @"C:\Users\firat\source\repos\MLDersleri\GitHubLabeler\Data";//Veri setlerinin yer aldığı temel dizinin yolu.
string DataSetRelativePath = $"{BaseDatasetsRelativePath}/corefx-issues-train.tsv";//Eğitim veri setinin dosya yolunu belirler.
string DataSetLocation = GetAbsolutePath(DataSetRelativePath);//DataSetRelativePath'in mutlak yolunu alır.
string BaseModelsRelativePath = @"C:\Users\firat\source\repos\MLDersleri\GitHubLabeler\MLModels";//Modellerin yer aldığı temel dizinin yolu.
string ModelRelativePath = $"{BaseModelsRelativePath}/GitHubLabelerModel.zip";//Eğitim sonrası kaydedilecek modelin dosya yolunu belirler.
string ModelPath = GetAbsolutePath(ModelRelativePath);//ModelRelativePath'in mutlak yolunu alır.
//Aşağıda tanımladığımız metotlar;
SetupAppConfiguration();//Uygulama yapılandırmasını ayarlar.
BuildAndTrainModel(DataSetLocation, ModelPath, MyTrainerStrategy.OVAAveragedPerceptronTrainer);//Modeli oluşturur ve eğitir.
TestSingleLabelPrediction(ModelPath);//Eğitilen model ile tek bir örnek üzerinde tahmin yapar.
await PredictLabelsAndUpdateGitHub(ModelPath);//GitHub'dan sorunları alır, tahmin eder ve etiketler.
//Aşağıda tanımladığımız metotlar.
ConsoleHelper.ConsolePressAnyKey();//Konsolda kullanıcıdan bir tuşa basmasını bekler.

void BuildAndTrainModel(string DataSetLocation, string ModelPath, MyTrainerStrategy selectedStrategy)
{
    MLContext mlContext = new MLContext(seed: 1);
    //EstimatorChain<ColumnConcatenatingTransformer>
    IDataView trainingDataView = mlContext.Data.LoadFromTextFile<GitHubIssue>(DataSetLocation, hasHeader: true, separatorChar: '\t', allowSparse: false);
    //Veri setini yükler.
    EstimatorChain<ColumnConcatenatingTransformer> dataProcessPipeline = mlContext.Transforms.Conversion//Veriyi dönüştürme ve özellik çıkarma işlemlerini tanımlar.
        .MapValueToKey(outputColumnName: "Label", inputColumnName: nameof(GitHubIssue.Area))//Area alanını sayısal bir anahtara dönüştürür.
        .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "TitleFeaturized", inputColumnName: nameof(GitHubIssue.Title)))//Title alanını sayısal özelliklere dönüştürür.
        .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "DescriptionFeaturized", inputColumnName: nameof(GitHubIssue.Description)))//Description alanını sayısal özelliklere dönüştürür.
        .Append(mlContext.Transforms.Concatenate(outputColumnName: "Features", "TitleFeaturized", "DescriptionFeaturized"))//TitleFeaturized ve DescriptionFeaturized özelliklerini birleştirir.
        .AppendCacheCheckpoint(mlContext);////Ara sonuçları önbelleğe alır.

    ConsoleHelper.PeekDataViewInConsole(mlContext, trainingDataView, dataProcessPipeline, 2);//Console Helper metodu.

    IEstimator<ITransformer> trainer = null;//Eğitici algoritmayı seçer ve ayarlar.

    switch (selectedStrategy)
    {
        case MyTrainerStrategy.SdcaMultiClassTrainer://SDCA algoritmasını kullanır.
            trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features");
            break;
        case MyTrainerStrategy.OVAAveragedPerceptronTrainer://Averaged Perceptron algoritmasını kullanır.
            {
                AveragedPerceptronTrainer averagedPerceptronBinaryTrainer =
                    mlContext.BinaryClassification.Trainers.AveragedPerceptron("Label", "Features", numberOfIterations: 10);
                trainer = mlContext.MulticlassClassification.Trainers.OneVersusAll(averagedPerceptronBinaryTrainer);
                break;
            }
        default:
            break;
    }

    //trainingPipeline: Eğitim hattını oluşturur.
    EstimatorChain<KeyToValueMappingTransformer> trainingPipeline = dataProcessPipeline
        .Append(trainer)
        .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));//Tahmin edilen etiketi orijinal metin değerine dönüştürür.

    Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");

    IReadOnlyList<TrainCatalogBase.CrossValidationResult<MulticlassClassificationMetrics>> crossValidationResults = mlContext.MulticlassClassification
        .CrossValidate(data: trainingDataView, estimator: trainingPipeline, numberOfFolds: 6, labelColumnName: "Label");//Modeli çapraz doğrulama ile değerlendirir.

    ConsoleHelper.PrintMulticlassClassificationFoldsAverageMetrics(trainer.ToString(), crossValidationResults);//Console Helper Metodu.

    Console.WriteLine("=============== Training the model ===============");//Yazı.

    TransformerChain<KeyToValueMappingTransformer> trainedModel = trainingPipeline.Fit(trainingDataView);//Eğitilmiş modeli oluşturur.

    GitHubIssue issue = new GitHubIssue()
    {
        ID = "Any-ID",
        Title = "WebSockets communication is slow in my machine",
        Description = "The WebSockets communication used under the covers by SignalR looks like is going slow in my development machine.."
    };

    //predEngine: Tahmin motoru oluşturur.
    PredictionEngine<GitHubIssue, GitHubIssuePrediction> predEngine = mlContext.Model.CreatePredictionEngine<GitHubIssue, GitHubIssuePrediction>(trainedModel);
    
    GitHubIssuePrediction prediction = predEngine.Predict(issue);//Tek bir örnek üzerinde tahmin yapar.

    Console.WriteLine($"=============== Single Prediction just-trained-model - Result: {prediction.Area} ===============");
    Console.WriteLine("=============== Saving the model to a file ===============");
    
    mlContext.Model.Save(trainedModel, trainingDataView.Schema, ModelPath);//Modeli dosyaya kaydeder.

    ConsoleHelper.ConsoleWriteHeader("Training process finalized");//Yazı.
}

void TestSingleLabelPrediction(string modelFilePathName)//Eğitilmiş model ile tek bir örnek üzerinde tahmin yapar.
{
    Labeler labeler = new Labeler(modelPath: ModelPath);
    labeler.TestPredictionForSingleIssue();
    //labeler: Labeler sınıfından bir nesne oluşturur ve tahmin yapar.
}

async Task PredictLabelsAndUpdateGitHub(string ModelPath)//GitHub'dan sorunları alır, tahmin eder ve etiketler.
{
    Console.WriteLine(".............Retrieving Issues from GITHUB repo, predicting label/s and assigning predicted label/s......");

    string token = Extensions.Configuration["GitHubToken"];//GitHub token'ını alır.
    string repoOwner = Extensions.Configuration["GitHubRepoOwner"];//Repo sahibini alır.
    string repoName = Extensions.Configuration["GitHubRepoName"];//Repo adını alır.

    if (string.IsNullOrEmpty(token) || token == "YOUR - GUID - GITHUB - TOKEN" ||
        string.IsNullOrEmpty(repoOwner) || repoOwner == "YOUR-REPO-USER-OWNER-OR-ORGANIZATION" ||
        string.IsNullOrEmpty(repoName) || repoName == "YOUR-REPO-SINGLE-NAME")
    {
        Console.Error.WriteLine();
        Console.Error.WriteLine("Error: please configure the credentials in the appsettings.json file");
        Console.ReadLine();
        return;
    }

    Labeler labeler = new Labeler(ModelPath, repoOwner, repoName, token);
    await labeler.LabelAllNewIssuesInGitHubRepo();
    //Labeler: Labeler sınıfından bir nesne oluşturur ve tahmin yapar.
    Console.WriteLine("Labeling completed");//Yazı.
    Console.ReadLine();//Kullanıcıdan input bekler.
}

void SetupAppConfiguration()//Uygulama yapılandırmasını ayarlar.
{
    //ConfigurationBuilder: Yapılandırma oluşturucu nesnesi oluşturur.
    IConfigurationBuilder builder = new ConfigurationBuilder().SetBasePath(Directory.GetCurrentDirectory()).AddJsonFile("appsettings.json");
    //SetBasePath: Temel dizini ayarlar.//AddJsonFile: appsettings.json dosyasını ekler.
    Extensions.Configuration = builder.Build();//Extensions.Configuration: Yapılandırma nesnesini oluşturur.
}

string GetAbsolutePath(string relativePath)//GetAbsolutePath: Göreli yolu mutlak yola dönüştürür.
{
    FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);//_dataRoot: Programın yürütüldüğü dosyanın bilgilerini alır.
    string assemblyFolderPath = _dataRoot.Directory.FullName;//Dizin yolunu alır.
    string fullPath = Path.Combine(assemblyFolderPath, relativePath);//Göreli yolu mutlak yola dönüştürmek için birleştirir.
    return fullPath;//Mutlak yolu döndürür.
}
