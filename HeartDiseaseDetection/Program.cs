using HeartDiseaseDetection.DataStructures;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using PLplot;

string BaseDatasetsRelativePath = @"C:\Users\firat\source\repos\MLDersleri\HeartDiseaseDetection\Data";
string TrainDataRelativePath = $"{BaseDatasetsRelativePath}/HeartTraining.csv";
string TestDataRelativePath = $"{BaseDatasetsRelativePath}/HeartTest.csv";
string TrainDataPath = GetAbsolutePath(TrainDataRelativePath);
string TestDataPath = GetAbsolutePath(TestDataRelativePath);
string BaseModelsRelativePath = @"C:\Users\firat\source\repos\MLDersleri\HeartDiseaseDetection\MLModels";
string ModelRelativePath = $"{BaseModelsRelativePath}/HeartClassification.zip";
string ModelPath = GetAbsolutePath(ModelRelativePath);
//Bu satırlar, eğitim ve test veri setlerinin yolunu, model dosyasının yolu ve diğer ilgili dosya yollarını tanımlar.
MLContext mlContext = new MLContext();
//MLContext nesnesi oluşturulur.
BuildTrainEvaluateAndSaveModel(mlContext);
//Eğitim, değerlendirme ve modelin kaydedilmesi işlemlerini yapan BuildTrainEvaluateAndSaveModel fonksiyonu çağrılır.
TestPrediction(mlContext);
//Eğitilen modelin üzerinde test yapmak için TestPrediction fonksiyonu çağrılır.
Console.WriteLine("=============== End of process, hit any key to finish ===============");
Console.ReadKey();
//İşlem sonunda kullanıcıya mesaj verilir ve herhangi bir tuşa basması beklenir.

//BuildTrainEvaluateAndSaveModel(): Eğitim veri setini yükler, modeli oluşturur, eğitir, test veri seti üzerinde değerlendirir ve sonuçları yazdırır. Ayrıca eğitilen modeli belirtilen yola kaydeder.

void BuildTrainEvaluateAndSaveModel(MLContext mlContext)
{
    IDataView trainingDataView = mlContext.Data.LoadFromTextFile<HeartData>(TrainDataPath, hasHeader: true, separatorChar: ';');
    //HeartData sınıfına uygun olarak CSV dosyasından veriyi yükler.
    //TrainDataPath: Eğitim verisinin dosya yolu.
    //hasHeader: true: CSV dosyasının bir başlık satırı olduğunu belirtir.
    //separatorChar: ';': CSV dosyasındaki sütunların ayrılması için kullanılan karakter.
    IDataView testDataView = mlContext.Data.LoadFromTextFile<HeartData>(TestDataPath, hasHeader: true, separatorChar: ';');
    //Yukarıdaki eğitim veri yükleme satırının aynısı, bu sefer test verisi için yapılır.
    EstimatorChain<BinaryPredictionTransformer<CalibratedModelParametersBase<FastTreeBinaryModelParameters, PlattCalibrator>>> pipeline = mlContext.Transforms
        .Concatenate("Features", "Age", "Sex", "Cp", "TrestBps", "Chol", "Fbs", "RestEcg", "Thalac", "Exang", "OldPeak", "Slope", "Ca", "Thal")//
        .Append(mlContext.BinaryClassification.Trainers.FastTree(labelColumnName: "Label", featureColumnName: "Features"));
    //Concatenate():Modelde kullanılacak özellikleri birleştirir ve "Features" adında bir sütun oluşturur.
    //Append(): FastTree() algoritmasıyla ikili sınıflandırma yapacak bir model eğitir.
    //labelColumnName: "Label": Hedef sütunun (etiket) adı.
    //featureColumnName: "Features": Özellik sütununun adı.
    Console.WriteLine("=============== Training the model ===============");
    //Modelin eğitim aşamasının başladığını belirtir.
    ITransformer trainedModel = pipeline.Fit(trainingDataView);
    //Eğitim verisiyle modeli eğitir ve eğitimli modeli (trainedModel) döner.
    Console.WriteLine("");
    Console.WriteLine("");
    Console.WriteLine("=============== Finish the train model. Push Enter ===============");
    Console.WriteLine("");
    Console.WriteLine("");
    Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
    //Model eğitiminin bittiğini ve değerlendirme aşamasının başladığını belirtir.
    IDataView predictions = trainedModel.Transform(testDataView);
    //Test verisini kullanarak tahminler yapar.
    CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");
    //Test verisi üzerindeki tahminleri değerlendirir ve modelin performans metriklerini hesaplar.
    //data: predictions: Tahminlerin yer aldığı veri seti.
    //labelColumnName: "Label": Gerçek etiket sütununun adı.
    //scoreColumnName: "Score": Tahmin edilen puan sütununun adı.
    Console.WriteLine("");
    Console.WriteLine("");
    Console.WriteLine($"************************************************************");
    Console.WriteLine($"*       Metrics for {trainedModel.ToString()} binary classification model      ");
    Console.WriteLine($"*-----------------------------------------------------------");
    Console.WriteLine($"*       Accuracy: {metrics.Accuracy:P2}");//Doğruluk.
    Console.WriteLine($"*       Area Under Roc Curve:      {metrics.AreaUnderRocCurve:P2}");//ROC eğrisi altındaki alan.
    Console.WriteLine($"*       Area Under PrecisionRecall Curve:  {metrics.AreaUnderPrecisionRecallCurve:P2}");//Precision-Recall eğrisi altındaki alan.
    Console.WriteLine($"*       F1Score:  {metrics.F1Score:P2}");//F1 skoru.
    Console.WriteLine($"*       LogLoss:  {metrics.LogLoss:#.##}");//Logaritmik kayıp.
    Console.WriteLine($"*       LogLossReduction:  {metrics.LogLossReduction:#.##}");//Logaritmik kayıp azalması.
    Console.WriteLine($"*       PositivePrecision:  {metrics.PositivePrecision:#.##}");//Pozitif sınıf için kesinlik.
    Console.WriteLine($"*       PositiveRecall:  {metrics.PositiveRecall:#.##}");//Pozitif sınıf için duyarlılık.
    Console.WriteLine($"*       NegativePrecision:  {metrics.NegativePrecision:#.##}");//Negatif sınıf için kesinlik.
    Console.WriteLine($"*       NegativeRecall:  {metrics.NegativeRecall:P2}");//Negatif sınıf için duyarlılık.
    Console.WriteLine($"************************************************************");
    Console.WriteLine("");
    Console.WriteLine("");
    Console.WriteLine("=============== Saving the model to a file ===============");
    //Modelin kaydedilme aşamasının başladığını belirtir.
    mlContext.Model.Save(trainedModel, trainingDataView.Schema, ModelPath);
    //Eğitilmiş modeli dosyaya kaydeder. trainedModel: Kaydedilecek model. trainingDataView.Schema: Eğitim verisinin şeması. ModelPath: Modelin kaydedileceği dosya yolu.
    Console.WriteLine("");
    Console.WriteLine("");
    Console.WriteLine("=============== Model Saved ============= ");
    //Modelin başarıyla kaydedildiğini belirtir.
}

//Bu fonksiyon, eğitimli modeli yükler, bir tahmin motoru oluşturur ve örnek veri listesi (HeartSampleData.heartDataList) üzerindeki her bir veri için tahmin yapar. Tahmin sonuçlarını (yaş, cinsiyet, vs.) ve tahmin edilen etiketi (hastalık var/yok) konsola yazdırır.
void TestPrediction(MLContext mlContext)
{
    ITransformer trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);
    //Eğitilmiş modeli belirtilen dosya yolundan yükler. ModelPath: Modelin kaydedildiği dosya yolu.
    //modelInputSchema: Modelin giriş verisi şeması(bu değişken daha sonra kullanılmaz, sadece modelin doğru yüklendiğinden emin olmak için kullanılır).
    PredictionEngine<HeartData, HeartPrediction> predictionEngine = mlContext.Model.CreatePredictionEngine<HeartData, HeartPrediction>(trainedModel);
    //HeartData türünde giriş verisi alıp HeartPrediction türünde tahmin üreten bir PredictionEngine oluşturur.
    //HeartData: Giriş verisi tipi(modelin kullanacağı veri).
    //HeartPrediction: Tahmin sonuçlarının tipi (modelin üreteceği sonuçlar).
    foreach (HeartData heartData in HeartSampleData.heartDataList)
    {//HeartData sınıfındaki örnek veri listesindeki her bir veri için tahmin yapar.
        HeartPrediction prediction = predictionEngine.Predict(heartData);
        //Her bir heartData örneği için tahmin yapar ve sonucu HeartPrediction tipindeki prediction değişkenine atar. Alttada bunları Konsola yazdırır.
        Console.WriteLine($"=============== Single Prediction  ===============");
        Console.WriteLine($"Age: {heartData.Age} ");
        Console.WriteLine($"Sex: {heartData.Sex} ");
        Console.WriteLine($"Cp: {heartData.Cp} ");
        Console.WriteLine($"TrestBps: {heartData.TrestBps} ");
        Console.WriteLine($"Chol: {heartData.Chol} ");
        Console.WriteLine($"Fbs: {heartData.Fbs} ");
        Console.WriteLine($"RestEcg: {heartData.RestEcg} ");
        Console.WriteLine($"Thalac: {heartData.Thalac} ");
        Console.WriteLine($"Exang: {heartData.Exang} ");
        Console.WriteLine($"OldPeak: {heartData.OldPeak} ");
        Console.WriteLine($"Slope: {heartData.Slope} ");
        Console.WriteLine($"Ca: {heartData.Ca} ");
        Console.WriteLine($"Thal: {heartData.Thal} ");
        Console.WriteLine($"Prediction Value: {prediction.Prediction} ");
        Console.WriteLine($"Prediction: {(prediction.Prediction ? "A disease could be present" : "Not present disease")} ");
        Console.WriteLine($"Probability: {prediction.Probability} ");
        Console.WriteLine($"==================================================");
        Console.WriteLine("");
        Console.WriteLine("");
    }
}

//GetAbsolutePath(): Bu yardımcı fonksiyon, verilen göreceli yolu kullanarak tam bir dosya yolu oluşturur. Bu, veri setlerinin ve model dosyasının yerini belirlemek için kullanılır.
string GetAbsolutePath(string relativePath)
{
    FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
    string assemblyFolderPath = _dataRoot.Directory.FullName;
    string fullPath = Path.Combine(assemblyFolderPath, relativePath);
    return fullPath;
}
