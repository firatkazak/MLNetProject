using BikeSharingDemand;
using BikeSharingDemand.DataStructures;
using Common;
using Microsoft.ML;
using Microsoft.ML.Data;

//Yol ve Konum Tanımlamaları;
string ModelsLocation = @"C:\Users\firat\source\repos\MLDersleri\BikeSharingDemand\MLModels";
string DatasetsLocation = @"C:\Users\firat\source\repos\MLDersleri\BikeSharingDemand\Data";
string TrainingDataRelativePath = $"{DatasetsLocation}/hour_train.csv";
string TestDataRelativePath = $"{DatasetsLocation}/hour_test.csv";
string TrainingDataLocation = GetAbsolutePath(TrainingDataRelativePath);
string TestDataLocation = GetAbsolutePath(TestDataRelativePath);

//ML.NET işlemleri için MLContext nesnesi oluşturulur. seed parametresi, rastgele sayı üreticisinin başlangıç durumunu belirtir.
MLContext mlContext = new MLContext(seed: 0);

//Veri Yükleme: LoadFromTextFile metoduyla CSV dosyaları yüklenir. DemandObservation sınıfı veri yapıları kullanılarak yükleme yapılır.
IDataView trainingDataView = mlContext.Data.LoadFromTextFile<DemandObservation>(path: TrainingDataLocation, hasHeader: true, separatorChar: ',');
IDataView testDataView = mlContext.Data.LoadFromTextFile<DemandObservation>(path: TestDataLocation, hasHeader: true, separatorChar: ',');

//Veri İşleme Pipeline Oluşturma: Veri işleme için bir pipeline oluşturulur. Concatenate ile belirtilen özellikler birleştirilir ve AppendCacheCheckpoint ile veri işleme işlemi önbelleğe alınır.
EstimatorChain<ColumnConcatenatingTransformer> dataProcessPipeline = mlContext.Transforms
    .Concatenate("Features",
                                         nameof(DemandObservation.Season), nameof(DemandObservation.Year), nameof(DemandObservation.Month),
                                         nameof(DemandObservation.Hour), nameof(DemandObservation.Holiday), nameof(DemandObservation.Weekday),
                                         nameof(DemandObservation.WorkingDay), nameof(DemandObservation.Weather), nameof(DemandObservation.Temperature),
                                         nameof(DemandObservation.NormalizedTemperature), nameof(DemandObservation.Humidity), nameof(DemandObservation.Windspeed))
    .AppendCacheCheckpoint(mlContext);

//ConsoleHelper metotları
ConsoleHelper.PeekDataViewInConsole(mlContext, trainingDataView, dataProcessPipeline, 10);
ConsoleHelper.PeekVectorColumnDataInConsole(mlContext, "Features", trainingDataView, dataProcessPipeline, 10);

//Bu kısım, farklı regresyon modellerini (FastTree, Poisson, SDCA, FastTreeTweedie) tanımlayan bir dizi oluşturuyor.
(string name, IEstimator<ITransformer> value)[] regressionLearners =
{
                ("FastTree", mlContext.Regression.Trainers.FastTree()),
                ("Poisson", mlContext.Regression.Trainers.LbfgsPoissonRegression()),
                ("SDCA", mlContext.Regression.Trainers.Sdca()),
                ("FastTreeTweedie", mlContext.Regression.Trainers.FastTreeTweedie()),
};

//Regresyon Modellerinin Eğitimi ve Değerlendirilmesi;
foreach ((string name, IEstimator<ITransformer> value) trainer in regressionLearners)
{
    Console.WriteLine("=============== Training the current model ===============");
    EstimatorChain<ITransformer> trainingPipeline = dataProcessPipeline.Append(trainer.value);
    TransformerChain<ITransformer> trainedModel = trainingPipeline.Fit(trainingDataView);

    Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
    IDataView predictions = trainedModel.Transform(testDataView);
    RegressionMetrics metrics = mlContext.Regression.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");
    ConsoleHelper.PrintRegressionMetrics(trainer.value.ToString(), metrics);

    string modelRelativeLocation = $"{ModelsLocation}/{trainer.name}Model.zip";
    string modelPath = GetAbsolutePath(modelRelativeLocation);
    mlContext.Model.Save(trainedModel, trainingDataView.Schema, modelPath);
    Console.WriteLine("The model is saved to {0}", modelPath);
}

//Model Tahmini ve Görselleştirme:
foreach ((string name, IEstimator<ITransformer> value) learner in regressionLearners)
{
    string modelRelativeLocation = $"{ModelsLocation}/{learner.name}Model.zip";
    string modelPath = GetAbsolutePath(modelRelativeLocation);
    ITransformer trainedModel = mlContext.Model.Load(modelPath, out DataViewSchema modelInputSchema);

    PredictionEngine<DemandObservation, DemandPrediction> predEngine = mlContext.Model.CreatePredictionEngine<DemandObservation, DemandPrediction>(trainedModel);

    Console.WriteLine($"================== Visualize/test 10 predictions for model {learner.name}Model.zip ==================");
    ModelScoringTester.VisualizeSomePredictions(mlContext, learner.name, TestDataLocation, predEngine, 10);
}

//ConsoleHelper metodu;
ConsoleHelper.ConsolePressAnyKey();

//Path metodu;
string GetAbsolutePath(string relativePath)
{
    FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
    string assemblyFolderPath = _dataRoot.Directory.FullName;
    string fullPath = Path.Combine(assemblyFolderPath, relativePath);
    return fullPath;
}
