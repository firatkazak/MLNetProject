using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;
using PowerAnomalyDetection.DataStructures;

//Yollar;
string DatasetsRelativePath = @"C:\Users\firat\source\repos\MLDersleri\PowerAnomalyDetection\Data";
string TrainingDatarelativePath = $"{DatasetsRelativePath}/power-export_min.csv";
string TrainingDataPath = GetAbsolutePath(TrainingDatarelativePath);
string BaseModelsRelativePath = @"C:\Users\firat\source\repos\MLDersleri\PowerAnomalyDetection\MLModels";
string ModelRelativePath = $"{BaseModelsRelativePath}/PowerAnomalyDetectionModel.zip";
string ModelPath = GetAbsolutePath(ModelRelativePath);

//MLContext, ML.NET işlemleri için ana giriş noktasıdır ve bu, rasgele sayı üretimi için bir tohum değeriyle başlatılır;
MLContext mlContext = new MLContext(seed: 0);

//Veriler, MeterData sınıfına göre power-export_min.csv dosyasından yüklenir. Verinin başlıkları olduğu belirtilir ve sütunlar virgülle ayrılmıştır;
IDataView dataView = mlContext.Data.LoadFromTextFile<MeterData>(TrainingDataPath, separatorChar: ',', hasHeader: true);

//Modelin eğitilmesi için metot çağrılır;
BuildTrainModel(mlContext, dataView);

//Anormalliklerin tespit edilmesi için metot çağrılır;
DetectAnomalies(mlContext, dataView);

//Konsol işleri;
Console.WriteLine("\nPress any key to exit");
Console.Read();

//
void BuildTrainModel(MLContext mlContext, IDataView dataView)
{//Zaman serisi anomali tespiti için model parametreleri belirlenir:
    const int PValueSize = 30;//P-değer geçmişi uzunluğu.
    const int SeasonalitySize = 30;//Mevsimsel dönem uzunluğu.
    const int TrainingSize = 90;//Eğitim pencere boyutu.
    const double ConfidenceInterval = 98;//ConfidenceInterval: Güven aralığı yüzdesi.

    string outputColumnName = nameof(SpikePrediction.Prediction);//Girdi sütun adları belirlenir.
    string inputColumnName = nameof(MeterData.ConsumptionDiffNormalized);//çıktı sütun adları belirlenir.

    //SSA (Singular Spectrum Analysis) kullanılarak spike (ani artış) tespiti için bir eğitim boru hattı oluşturulur;
    SsaSpikeEstimator trainigPipeLine = mlContext.Transforms.DetectSpikeBySsa
        (
        outputColumnName,//Bu parametre, modelin tahmin sonuçlarını hangi sütunda saklayacağını belirler. Burada nameof(SpikePrediction.Prediction) kullanılmıştır, yani tahminler Prediction sütununda saklanacaktır.
        inputColumnName,//Bu parametre, modelin giriş verilerini hangi sütundan alacağını belirler. Burada nameof(MeterData.ConsumptionDiffNormalized) kullanılmıştır, yani giriş verileri ConsumptionDiffNormalized sütunundan alınacaktır.
        confidence: ConfidenceInterval,//Bu parametre, güven aralığı yüzdesini belirler. Güven aralığı, tahminlerin güvenilirliğini ifade eder ve burada %98 olarak ayarlanmıştır.
        pvalueHistoryLength: PValueSize,//Bu parametre, P-değeri hesaplaması için kullanılacak geçmiş verilerin uzunluğunu belirler. P-değeri, tahmin edilen değerin anomali olup olmadığını belirlemek için kullanılır. Burada 30 olarak ayarlanmıştır.
        trainingWindowSize: TrainingSize,//Bu parametre, modelin eğitimi sırasında kullanılacak veri pencere boyutunu belirler. Eğitim penceresi, modelin geçmiş verilere dayalı olarak nasıl tahmin yapacağını öğrenmesini sağlar. Burada 90 olarak ayarlanmıştır.
        seasonalityWindowSize: SeasonalitySize//Bu parametre, mevsimsellik döneminin uzunluğunu belirler. Mevsimsellik, verilerde tekrar eden kalıpların tespit edilmesini sağlar. Burada 30 olarak ayarlanmıştır.
        );

    //Model, veri kümesi üzerinde eğitilir;
    ITransformer trainedModel = trainigPipeLine.Fit(dataView);

    //Eğitilen model, belirtilen yola kaydedilir;
    mlContext.Model.Save(trainedModel, dataView.Schema, ModelPath);

    //Modelin kaydedildiği yol konsola yazdırılır;
    Console.WriteLine("The model is saved to {0}", ModelPath);
    Console.WriteLine("");
}

//
void DetectAnomalies(MLContext mlContext, IDataView dataView)
{
    //Kaydedilmiş model yüklenir;
    ITransformer trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);

    //Model, veri kümesine uygulanarak tahminler oluşturulur;
    IDataView transformedData = trainedModel.Transform(dataView);

    //Tahmin sonuçları, SpikePrediction sınıfına göre enumerable (numaralandırılabilir) bir yapıya dönüştürülür;
    IEnumerable<SpikePrediction> predictions = mlContext.Data.CreateEnumerable<SpikePrediction>(transformedData, false);

    //Veri kümesindeki ConsumptionDiffNormalized ve time sütunları alınır ve dizilere dönüştürülür.
    float[] colCDN = dataView.GetColumn<float>("ConsumptionDiffNormalized").ToArray();
    DateTime[] colTime = dataView.GetColumn<DateTime>("time").ToArray();

    //Anormalliklerin görüntülenmesi için başlıklar konsola yazdırılır.
    Console.WriteLine("======Displaying anomalies in the Power meter data=========");
    Console.WriteLine("Date              \tReadingDiff\tAlert\tScore\tP-Value");

    //Tahmin sonuçları üzerinden geçilir ve eğer anormallik (spike) tespit edilmişse, arka plan rengi değiştirerek sonuçlar konsola yazdırılır;
    int i = 0;
    foreach (SpikePrediction p in predictions)
    {
        if (p.Prediction[0] == 1)
        {
            Console.BackgroundColor = ConsoleColor.DarkYellow;
            Console.ForegroundColor = ConsoleColor.Black;
        }
        Console.WriteLine("{0}\t{1:0.0000}\t{2:0.00}\t{3:0.00}\t{4:0.00}", colTime[i], colCDN[i], p.Prediction[0], p.Prediction[1], p.Prediction[2]);
        Console.ResetColor();
        i++;
    }
}

//Yol alma metodu;
string GetAbsolutePath(string relativeDatasetPath)
{
    FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
    string assemblyFolderPath = _dataRoot.Directory.FullName;
    string fullPath = Path.Combine(assemblyFolderPath, relativeDatasetPath);
    return fullPath;
}
