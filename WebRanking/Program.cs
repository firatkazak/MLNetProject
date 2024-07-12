using Microsoft.ML;
using System.Net;
using WebRanking.DataStructures;
using WebRanking.Common;

//Yollar;
string AssetsPath = @"C:\Users\firat\source\repos\MLDersleri\WebRanking\Assets";
string TrainDatasetUrl = "https://aka.ms/mlnet-resources/benchmarks/MSLRWeb10KTrain720kRows.tsv";
string ValidationDatasetUrl = "https://aka.ms/mlnet-resources/benchmarks/MSLRWeb10KValidate240kRows.tsv";
string TestDatasetUrl = "https://aka.ms/mlnet-resources/benchmarks/MSLRWeb10KTest240kRows.tsv";
string InputPath = Path.Combine(AssetsPath, "Input");
string OutputPath = Path.Combine(AssetsPath, "Output");
string TrainDatasetPath = Path.Combine(InputPath, "MSLRWeb10KTrain720kRows.tsv");
string ValidationDatasetPath = Path.Combine(InputPath, "MSLRWeb10KValidate240kRows.tsv");
string TestDatasetPath = Path.Combine(InputPath, "MSLRWeb10KTest240kRows.tsv");
string ModelPath = Path.Combine(OutputPath, "RankingModel.zip");

//Rastgele işlemler için sabit bir tohum değeri kullanarak bir MLContext oluşturulur. Bu, işlemlerin tekrarlanabilirliğini sağlar;
MLContext mlContext = new MLContext(seed: 0);

try
{
    PrepareData(InputPath, OutputPath, TrainDatasetPath, TrainDatasetUrl, TestDatasetUrl, TestDatasetPath, ValidationDatasetUrl, ValidationDatasetPath);

    //Eğitim verileri yüklenir.
    IDataView trainData = mlContext.Data.LoadFromTextFile<SearchResultData>(TrainDatasetPath, separatorChar: '\t', hasHeader: true);
    //CreatePipeline metodu ile veri işleme ve model eğitimi için bir pipeline oluşturulur.
    IEstimator<ITransformer> pipeline = CreatePipeline(mlContext, trainData);

    Console.WriteLine("===== Train the model on the training dataset =====\n");
    //Pipeline kullanılarak model eğitilir.
    ITransformer model = pipeline.Fit(trainData);

    Console.WriteLine("===== Evaluate the model's result quality with the validation data =====\n");
    //Model, doğrulama veri seti ile değerlendirilir.
    IDataView validationData = mlContext.Data.LoadFromTextFile<SearchResultData>(ValidationDatasetPath, separatorChar: '\t', hasHeader: false);
    EvaluateModel(mlContext, model, validationData);

    //Eğitim ve doğrulama verileri birleştirilir...
    IEnumerable<SearchResultData> validationDataEnum = mlContext.Data.CreateEnumerable<SearchResultData>(validationData, false);
    IEnumerable<SearchResultData> trainDataEnum = mlContext.Data.CreateEnumerable<SearchResultData>(trainData, false);
    IEnumerable<SearchResultData> trainValidationDataEnum = validationDataEnum.Concat<SearchResultData>(trainDataEnum);
    IDataView trainValidationData = mlContext.Data.LoadFromEnumerable<SearchResultData>(trainValidationDataEnum);
    //ve model yeniden eğitilir.
    Console.WriteLine("===== Train the model on the training + validation dataset =====\n");
    model = pipeline.Fit(trainValidationData);

    //Model, test veri seti ile değerlendirilir.
    Console.WriteLine("===== Evaluate the model's result quality with the testing data =====\n");
    IDataView testData = mlContext.Data.LoadFromTextFile<SearchResultData>(TestDatasetPath, separatorChar: '\t', hasHeader: false);
    EvaluateModel(mlContext, model, testData);

    //Eğitim, doğrulama ve test verileri birleştirilir ve model yeniden eğitilir;
    IEnumerable<SearchResultData> testDataEnum = mlContext.Data.CreateEnumerable<SearchResultData>(testData, false);
    IEnumerable<SearchResultData> allDataEnum = trainValidationDataEnum.Concat<SearchResultData>(testDataEnum);
    IDataView allData = mlContext.Data.LoadFromEnumerable<SearchResultData>(allDataEnum);
    Console.WriteLine("===== Train the model on the training + validation + test dataset =====\n");
    model = pipeline.Fit(allData);

    //ConsumeModel metodu ile model kaydedilir ve kullanılır;
    ConsumeModel(mlContext, model, ModelPath, testData);
}
catch (Exception e)
{
    Console.WriteLine(e.Message);
}

// Done yazar ve kullanıcıdan Input bekler.
Console.Write("Done!");
Console.ReadLine();

//PrepareData metodu ile gerekli veri setleri indirilir ve dosya yollarına kaydedilir;
void PrepareData(string inputPath, string outputPath, string trainDatasetPath, string trainDatasetUrl, string testDatasetUrl, string testDatasetPath, string validationDatasetUrl, string validationDatasetPath)
{
    Console.WriteLine("===== Prepare data =====\n");

    if (!Directory.Exists(outputPath))
    {
        Directory.CreateDirectory(outputPath);
    }

    if (!Directory.Exists(inputPath))
    {
        Directory.CreateDirectory(inputPath);
    }

    if (!File.Exists(trainDatasetPath))
    {
        Console.WriteLine("===== Download the train dataset - this may take several minutes =====\n");
        using (WebClient client = new WebClient())
        {
            client.DownloadFile(trainDatasetUrl, TrainDatasetPath);
        }
    }

    if (!File.Exists(validationDatasetPath))
    {
        Console.WriteLine("===== Download the validation dataset - this may take several minutes =====\n");
        using (WebClient client = new WebClient())
        {
            client.DownloadFile(validationDatasetUrl, validationDatasetPath);
        }
    }

    if (!File.Exists(testDatasetPath))
    {
        Console.WriteLine("===== Download the test dataset - this may take several minutes =====\n");
        using (WebClient client = new WebClient())
        {
            client.DownloadFile(testDatasetUrl, testDatasetPath);
        }
    }

    Console.WriteLine("===== Download is finished =====\n");
}

//Veri işleme ve model eğitimi için bir pipeline oluşturur.
IEstimator<ITransformer> CreatePipeline(MLContext mlContext, IDataView dataView)
{
    const string FeaturesVectorName = "Features";

    Console.WriteLine("===== Set up the trainer =====\n");

    string[] featureCols = dataView.Schema.AsQueryable().Select(s => s.Name).Where(c => c != nameof(SearchResultData.Label) && c != nameof(SearchResultData.GroupId)).ToArray();

    IEstimator<ITransformer> dataPipeline = mlContext.Transforms.Concatenate(FeaturesVectorName, featureCols)
        .Append(mlContext.Transforms.Conversion.MapValueToKey(nameof(SearchResultData.Label)))
        .Append(mlContext.Transforms.Conversion.Hash(nameof(SearchResultData.GroupId), nameof(SearchResultData.GroupId), numberOfBits: 20));

    IEstimator<ITransformer> trainer = mlContext.Ranking.Trainers
        .LightGbm(labelColumnName: nameof(SearchResultData.Label), featureColumnName: FeaturesVectorName, rowGroupColumnName: nameof(SearchResultData.GroupId));

    IEstimator<ITransformer> trainerPipeline = dataPipeline.Append(trainer);

    return trainerPipeline;
}

//Modelin performansını değerlendirir;
void EvaluateModel(MLContext mlContext, ITransformer model, IDataView data)
{
    IDataView predictions = model.Transform(data);

    Console.WriteLine("===== Use metrics for the data using NDCG@3 =====\n");

    ConsoleHelper.EvaluateMetrics(mlContext, predictions);
}

//Modeli kaydeder ve kullanarak tahmin yapar;
void ConsumeModel(MLContext mlContext, ITransformer model, string modelPath, IDataView data)
{
    Console.WriteLine("===== Save the model =====\n");

    mlContext.Model.Save(model, null, modelPath);

    Console.WriteLine("===== Consume the model =====\n");

    DataViewSchema predictionPipelineSchema;
    ITransformer predictionPipeline = mlContext.Model.Load(modelPath, out predictionPipelineSchema);

    IDataView predictions = predictionPipeline.Transform(data);

    IEnumerable<SearchResultPrediction> searchQueries = mlContext.Data.CreateEnumerable<SearchResultPrediction>(predictions, reuseRowObject: false);
    uint firstGroupId = searchQueries.First<SearchResultPrediction>().GroupId;
    IEnumerable<SearchResultPrediction> firstGroupPredictions = searchQueries.Take(100).Where(p => p.GroupId == firstGroupId).OrderByDescending(p => p.Score).ToList();

    ConsoleHelper.PrintScores(firstGroupPredictions);
}
