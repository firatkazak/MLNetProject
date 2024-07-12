using Common;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System.Drawing;

//Bu kısımda modelin ve verilerin yollarını belirleyen string değişkenler tanımlanıyor;
string BaseModelRelativePath = @"C:\Users\firat\source\repos\MLDersleri\MovieRecommender_Model\Model";
string ModelRelativePath = $"{BaseModelRelativePath}/model.zip";
string BaseDataSetRelativepath = @"C:\Users\firat\source\repos\MLDersleri\MovieRecommender_Model\Data";
string TrainingDataRelativePath = $"{BaseDataSetRelativepath}/ratings_train.csv";
string TestDataRelativePath = $"{BaseDataSetRelativepath}/ratings_test.csv";
string TrainingDataLocation = GetAbsolutePath(TrainingDataRelativePath);
string TestDataLocation = GetAbsolutePath(TestDataRelativePath);
string ModelPath = GetAbsolutePath(ModelRelativePath);
//GetAbsolutePath fonksiyonu ile bu yolların mutlak yolları elde ediliyor.

//Bu kısımda konsolda kullanılacak metin rengi tanımlanıyor;
Color color = Color.FromArgb(130, 150, 115);

//DataPrep metodu, orijinal veri setini eğitim ve test setlerine ayırmak için çağrılıyor;
DataPrep();

//MLContext nesnesi, ML.NET işlemlerini yürütmek için gereken ortamı sağlar;
MLContext mlContext = new MLContext();

//ratings_train.csv dosyasındaki veriler IDataView formatında yükleniyor. MovieRating sınıfı, bu verilerin yapısını belirliyor;
IDataView trainingDataView = mlContext.Data.LoadFromTextFile<MovieRating>(path: TrainingDataLocation, hasHeader: true, separatorChar: ',');

Console.WriteLine("=============== Reading Input Files ===============", color);
Console.WriteLine();

//Veriler bellek içi önbelleğe alınıyor, bu da işlemleri hızlandırıyor;
trainingDataView = mlContext.Data.Cache(trainingDataView);

Console.WriteLine("=============== Transform Data And Preview ===============", color);
Console.WriteLine();

//Bu kısımda, kullanıcı ve film ID'leri featurize ediliyor (sayısal değerlere dönüştürülüyor) ve ardından bu özellikler birleştiriliyor;
EstimatorChain<TransformerChain<ColumnConcatenatingTransformer>> dataProcessPipeline =
    mlContext.Transforms.Text.FeaturizeText(outputColumnName: "userIdFeaturized", inputColumnName: nameof(MovieRating.userId))
    .Append(mlContext.Transforms.Text.FeaturizeText(outputColumnName: "movieIdFeaturized", inputColumnName: nameof(MovieRating.movieId))
    .Append(mlContext.Transforms.Concatenate("Features", "userIdFeaturized", "movieIdFeaturized")));

//Veri işleme hattı uygulanmış verilerden örnekler konsolda görüntüleniyor;
ConsoleHelper.PeekDataViewInConsole(mlContext, trainingDataView, dataProcessPipeline, 10);

Console.WriteLine("=============== Training the model ===============", color);
Console.WriteLine();

//Bu kısımda, özellik mühendisliği hattına Field-Aware Factorization Machine (FAFM) sınıflandırıcı ekleniyor;
EstimatorChain<FieldAwareFactorizationMachinePredictionTransformer> trainingPipeLine =
    dataProcessPipeline.Append(mlContext.BinaryClassification.Trainers.FieldAwareFactorizationMachine(new string[] { "Features" }));

//Yukarıda eklenen model, eğitim verisi üzerinde eğitiliyor;
TransformerChain<FieldAwareFactorizationMachinePredictionTransformer> model = trainingPipeLine.Fit(trainingDataView);

Console.WriteLine("=============== Evaluating the model ===============", color);
Console.WriteLine();

//Test verisi yükleniyor;
IDataView testDataView = mlContext.Data.LoadFromTextFile<MovieRating>(path: TestDataLocation, hasHeader: true, separatorChar: ',');

//Yukarıda eklenen model üzerinde tahmin yapılıyor;
IDataView prediction = model.Transform(testDataView);

//Ardından, modelin doğruluk ve ROC eğrisi gibi metrikleri hesaplanıyor;
CalibratedBinaryClassificationMetrics metrics =
    mlContext.BinaryClassification.Evaluate(data: prediction, labelColumnName: "Label", scoreColumnName: "Score", predictedLabelColumnName: "PredictedLabel");

//Sonuç konsolda görüntüleniyor;
Console.WriteLine("Evaluation Metrics: acc:" + Math.Round(metrics.Accuracy, 2) + " AreaUnderRocCurve(AUC):" + Math.Round(metrics.AreaUnderRocCurve, 2), color);

Console.WriteLine("=============== Test a single prediction ===============", color);
Console.WriteLine();

//Burada tek bir kullanıcı-film çifti için tahmin yapılıyor ve...;
PredictionEngine<MovieRating, MovieRatingPrediction> predictionEngine = mlContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);

MovieRating testData = new MovieRating() { userId = "6", movieId = "10" };

MovieRatingPrediction movieRatingPrediction = predictionEngine.Predict(testData);

Console.WriteLine($"UserId:{testData.userId} with movieId: {testData.movieId} Score:{Sigmoid(movieRatingPrediction.Score)} and Label {movieRatingPrediction.PredictedLabel}", Color.YellowGreen);
Console.WriteLine();

Console.WriteLine("=============== Writing model to the disk ===============", color);
Console.WriteLine(); mlContext.Model.Save(model, trainingDataView.Schema, ModelPath);

Console.WriteLine("=============== Re-Loading model from the disk ===============", color);
Console.WriteLine();
//Sonuç konsolda görüntüleniyor.

//Eğitilmiş model diske kaydediliyor...;
ITransformer trainedModel;

//ve ardından yeniden yükleniyor;
using (FileStream stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
{
    trainedModel = mlContext.Model.Load(stream, out var modelInputSchema);
}

Console.WriteLine("Press any key ...");
Console.Read();
//Kullanıcıdan bir Input bekliyor.

//DataPrep metodu, ratings.csv dosyasını okuyup, her bir rating değerini 1 veya 0 (binary sınıflandırma) olarak yeniden düzenliyor ve ardından veriyi eğitim ve test setlerine ayırıyor;
void DataPrep()
{

    string[] dataset = File.ReadAllLines(@"C:\Users\firat\source\repos\MLDersleri\MovieRecommender_Model\Data\ratings.csv");

    string[] new_dataset = new string[dataset.Length];
    new_dataset[0] = dataset[0];

    for (int i = 1; i < dataset.Length; i++)
    {
        string line = dataset[i];
        string[] lineSplit = line.Split(',');
        double rating = Double.Parse(lineSplit[2]);
        rating = rating > 3 ? 1 : 0;
        lineSplit[2] = rating.ToString();
        string new_line = string.Join(',', lineSplit);
        new_dataset[i] = new_line;
    }

    dataset = new_dataset;
    int numLines = dataset.Length;

    IEnumerable<string> body = dataset.Skip(1);
    IEnumerable<string> sorted = body
        .Select(line => new { SortKey = Int32.Parse(line.Split(',')[3]), Line = line })
        .OrderBy(x => x.SortKey)
        .Select(x => x.Line);

    File.WriteAllLines(@"C:\Users\firat\source\repos\MLDersleri\MovieRecommender_Model\Data\ratings_train.csv", dataset.Take(1).Concat(sorted.Take((int)(numLines * 0.9))));
    File.WriteAllLines(@"C:\Users\firat\source\repos\MLDersleri\MovieRecommender_Model\Data\ratings_test.csv", dataset.Take(1).Concat(sorted.TakeLast((int)(numLines * 0.1))));
}

//Bu fonksiyon, bir değeri sigmoid fonksiyonuna göre dönüştürür ve genellikle tahmin skorlarını normalize etmek için kullanılır;
float Sigmoid(float x)
{
    return (float)(100 / (1 + Math.Exp(-x)));
}

//Bu fonksiyon, verilen relatif yolun mutlak yolunu döndürür;
string GetAbsolutePath(string relativeDatasetPath)
{
    FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
    string assemblyFolderPath = _dataRoot.Directory.FullName;
    string fullPath = Path.Combine(assemblyFolderPath, relativeDatasetPath);
    return fullPath;
}

//MovieRating sınıfı, veri setinin şemasını tanımlar.
public class MovieRating
{
    [LoadColumn(0)]
    public string userId;

    [LoadColumn(1)]
    public string movieId;

    [LoadColumn(2)]
    public bool Label;
}

//MovieRatingPrediction sınıfı tahmin sonuçlarını tutar.
public class MovieRatingPrediction
{
    public bool PredictedLabel;

    public float Score;
}
