using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Onnx;

//Yollar;
string currentDirectory = AppDomain.CurrentDomain.BaseDirectory;
string TRAIN_DATA_FILEPATH = Path.Combine(currentDirectory, @"C:\Users\firat\source\repos\MLDersleri\ONNXExport\Data\taxi-fare-train.csv");
string TEST_DATA_FILEPATH = Path.Combine(currentDirectory, @"C:\Users\firat\source\repos\MLDersleri\ONNXExport\Data\taxi-fare-test.csv");

//PrintMLNetScore: ML.NET modelinden tahmin edilen skorları ekrana yazdırmak için bir yardımcı metot tanımlar. values parametresi skorlardan oluşan bir koleksiyon olup, numRows parametresi kaç tane skoru yazdırmak istediğinizi belirtir;
void PrintMLNetScore(IEnumerable<ScoreValue> values, int numRows)
{
    Console.WriteLine("Predicted Scores with ML.NET model");
    foreach (var value in values.Take(numRows))
        Console.WriteLine("{0, -10} {1, -10}", "Score", value.Score);
}

//PrintOnnxScore: ONNX modelinden tahmin edilen skorları ekrana yazdırmak için bir yardımcı metot tanımlar. values parametresi skorlardan oluşan bir koleksiyon olup, numRows parametresi kaç tane skoru yazdırmak istediğinizi belirtir;
void PrintOnnxScore(IEnumerable<OnnxScoreValue> values, int numRows)
{
    Console.WriteLine("Predicted Scores with ONNX model");
    foreach (var value in values.Take(numRows))
        Console.WriteLine("{0, -10} {1, -10}", "Score", value.Score.GetItemOrDefault(0));
}

//ML.NET bağlamını başlatır. Bu bağlam, veriyi yüklemek, dönüştürmek, modeli eğitmek ve değerlendirmek için kullanılır;
MLContext mlContext = new MLContext();

//Eğitim ve test verilerini yükler. ModelInput sınıfını kullanarak CSV dosyalarının yapısını belirtir ve IDataView veri yapısına yükler;
IDataView trainingDataView = mlContext.Data.LoadFromTextFile<ModelInput>(path: TRAIN_DATA_FILEPATH, hasHeader: true, separatorChar: ',');
IDataView testDataView = mlContext.Data.LoadFromTextFile<ModelInput>(path: TEST_DATA_FILEPATH, hasHeader: true, separatorChar: ',');

//Veri işleme ardışık düzenini tanımlar. vendor_id ve payment_type kategorik sütunlarını one-hot encoding ile dönüştürür ve bu sütunları diğer özelliklerle birleştirir;
EstimatorChain<ColumnConcatenatingTransformer> dataProcessPipeline = mlContext.Transforms.Categorical
    .OneHotEncoding(new[] { new InputOutputColumnPair("vendor_id", "vendor_id"), new InputOutputColumnPair("payment_type", "payment_type") })
    .Append(mlContext.Transforms
    .Concatenate("Features", new[] { "vendor_id", "payment_type", "rate_code", "passenger_count", "trip_time_in_secs", "trip_distance" }));

//Stochastic Dual Coordinate Ascent (SDCA) regresyon algoritmasını kullanarak bir regresyon modeli oluşturur. fare_amount etiket sütununu ve Features özellik sütununu belirtir;
SdcaRegressionTrainer trainer = mlContext.Regression.Trainers.Sdca(labelColumnName: "fare_amount", featureColumnName: "Features");

//Veri işleme ardışık düzeni ile modeli eğitmek için eğitim ardışık düzenini oluşturur;
EstimatorChain<RegressionPredictionTransformer<LinearRegressionModelParameters>> trainingPipeline = dataProcessPipeline.Append(trainer);

//Eğitim verilerini kullanarak modeli eğitir;
ITransformer model = trainingPipeline.Fit(trainingDataView);

//Eğitilmiş modeli ONNX formatına dönüştürür ve belirtilen dosya yoluna kaydeder;
using (FileStream stream = File.Create("taxi-fare-model.onnx")) mlContext.Model.ConvertToOnnx(model, trainingDataView, stream);

//ONNX model dosyasının yolunu belirler;
string onnxModelPath = "taxi-fare-model.onnx";
//ONNX modelini yükleyip değerlendirmek için bir estimator oluşturur;
OnnxScoringEstimator onnxEstimator = mlContext.Transforms.ApplyOnnxModel(onnxModelPath);

//ONNX modelini kullanarak bir transformer oluşturur;
using OnnxTransformer onnxTransformer = onnxEstimator.Fit(trainingDataView);

//ML.NET modelini kullanarak test verilerini dönüştürür ve tahmin sonuçlarını elde eder;
IDataView output = model.Transform(testDataView);

//ONNX modelini kullanarak test verilerini dönüştürür ve tahmin sonuçlarını elde eder;
IDataView onnxOutput = onnxTransformer.Transform(testDataView);

//ML.NET modelinden elde edilen tahmin sonuçlarını ScoreValue sınıfına dönüştürür;
IEnumerable<ScoreValue> outScores = mlContext.Data.CreateEnumerable<ScoreValue>(output, reuseRowObject: false);

//ONNX modelinden elde edilen tahmin sonuçlarını OnnxScoreValue sınıfına dönüştürür;
IEnumerable<OnnxScoreValue> onnxOutScores = mlContext.Data.CreateEnumerable<OnnxScoreValue>(onnxOutput, reuseRowObject: false);

//ML.NET modelinden elde edilen ilk 5 tahmin sonucunu yazdırır;
PrintMLNetScore(outScores, 5);

//ONNX modelinden elde edilen ilk 5 tahmin sonucunu yazdırır;
PrintOnnxScore(onnxOutScores, 5);

//ModelInput: Eğitim ve test verilerinin her bir satırını temsil eder. Sütun adları ve yükleme indeksleri belirtilmiştir.
public class ModelInput
{
    [ColumnName("vendor_id"), LoadColumn(0)]
    public string Vendor_id { get; set; }


    [ColumnName("rate_code"), LoadColumn(1)]
    public float Rate_code { get; set; }


    [ColumnName("passenger_count"), LoadColumn(2)]
    public float Passenger_count { get; set; }


    [ColumnName("trip_time_in_secs"), LoadColumn(3)]
    public float Trip_time_in_secs { get; set; }


    [ColumnName("trip_distance"), LoadColumn(4)]
    public float Trip_distance { get; set; }


    [ColumnName("payment_type"), LoadColumn(5)]
    public string Payment_type { get; set; }


    [ColumnName("fare_amount"), LoadColumn(6)]
    public float Fare_amount { get; set; }

}

//ModelOutput: Modelin tahmin sonuçlarını temsil eder.
public class ModelOutput
{
    public float Score { get; set; }
}

//ScoreValue: ML.NET modelinden elde edilen tahmin sonuçlarını saklamak için kullanılan sınıf.
public class ScoreValue
{
    public float Score { get; set; }
}

//OnnxScoreValue: ONNX modelinden elde edilen tahmin sonuçlarını saklamak için kullanılan sınıf. VBuffer ONNX modellerinde kullanılan bir veri yapısıdır.
public class OnnxScoreValue
{
    public VBuffer<float> Score { get; set; }
}
