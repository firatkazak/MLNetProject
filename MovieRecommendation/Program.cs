using Microsoft.ML.Trainers;
using Microsoft.ML;
using MovieRecommendation.DataStructures;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Trainers.Recommender;

//Model, veri seti ve film verilerinin dosya yollarını tanımlar;
string ModelsRelativePath = @"C:\Users\firat\source\repos\MLDersleri\MovieRecommendation\MLModels";
string DatasetsRelativePath = @"C:\Users\firat\source\repos\MLDersleri\MovieRecommendation\Data\";
string TrainingDataRelativePath = $"{DatasetsRelativePath}/recommendation-ratings-train.csv";
string TestDataRelativePath = $"{DatasetsRelativePath}/recommendation-ratings-test.csv";
string MoviesDataLocation = $"{DatasetsRelativePath}/movies.csv";

//GetAbsolutePath fonksiyonu ile göreceli dosya yolları tam yol olarak alınır;
string TrainingDataLocation = GetAbsolutePath(TrainingDataRelativePath);
string TestDataLocation = GetAbsolutePath(TestDataRelativePath);
string ModelPath = GetAbsolutePath(ModelsRelativePath);

//
const float predictionuserId = 6;
const int predictionmovieId = 10;

//ML.NET işlemleri için MLContext nesnesi oluşturulur;
MLContext mlcontext = new MLContext();

//recommendation-ratings-train.csv dosyasından eğitim veri seti yüklenir.
IDataView trainingDataView = mlcontext.Data.LoadFromTextFile<MovieRating>(TrainingDataLocation, hasHeader: true, separatorChar: ',');

//Veri işleme için boru hattı oluşturulur, kullanıcı ve film ID'leri sayısal değerlere dönüştürülür;
EstimatorChain<ValueToKeyMappingTransformer> dataProcessingPipeline =
    mlcontext.Transforms.Conversion
    .MapValueToKey(outputColumnName: "userIdEncoded", inputColumnName: nameof(MovieRating.userId))
    .Append(mlcontext.Transforms.Conversion
    .MapValueToKey(outputColumnName: "movieIdEncoded", inputColumnName: nameof(MovieRating.movieId)));

//Matrix Factorization Trainer için seçenekler tanımlanır;
MatrixFactorizationTrainer.Options options = new MatrixFactorizationTrainer.Options();
options.MatrixColumnIndexColumnName = "userIdEncoded";
options.MatrixRowIndexColumnName = "movieIdEncoded";
options.LabelColumnName = "Label";
options.NumberOfIterations = 20;
options.ApproximationRank = 100;

//Veri işleme boru hattına Matrix Factorization Trainer eklenir.
EstimatorChain<MatrixFactorizationPredictionTransformer> trainingPipeLine = dataProcessingPipeline.Append(mlcontext.Recommendation().Trainers.MatrixFactorization(options));

//Eğitim veri seti üzerinde model eğitilir.
Console.WriteLine("=============== Training the model ===============");
ITransformer model = trainingPipeLine.Fit(trainingDataView);

//Test veri seti üzerinde model değerlendirilir ve performans metrikleri hesaplanır.
Console.WriteLine("=============== Evaluating the model ===============");
IDataView testDataView = mlcontext.Data.LoadFromTextFile<MovieRating>(TestDataLocation, hasHeader: true, separatorChar: ',');
IDataView prediction = model.Transform(testDataView);
RegressionMetrics metrics = mlcontext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");
Console.WriteLine("The model evaluation metrics RootMeanSquaredError:" + metrics.RootMeanSquaredError);

//Model kullanılarak belirli bir kullanıcı ve film için derecelendirme tahmini yapılır.
PredictionEngine<MovieRating, MovieRatingPrediction> predictionengine = mlcontext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);
MovieRatingPrediction movieratingprediction = predictionengine.Predict(new MovieRating()
{
    userId = predictionuserId,
    movieId = predictionmovieId
});

//Tahmin sonuçları konsola yazdırılır ve program sonlandırılır;
Movie movieService = new Movie();
Console.WriteLine("For userId:" + predictionuserId + " movie rating prediction (1 - 5 stars) for movie:" + movieService.Get(predictionmovieId).movieTitle + " is:" + Math.Round(movieratingprediction.Score, 1));

Console.WriteLine("=============== End of process, hit any key to finish ===============");
Console.ReadLine();

string GetAbsolutePath(string relativePath)//Path metodu.
{
    FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
    string assemblyFolderPath = _dataRoot.Directory.FullName;
    string fullPath = Path.Combine(assemblyFolderPath, relativePath);
    return fullPath;
}
