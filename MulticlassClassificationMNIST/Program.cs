using Common;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using MulticlassClassificationMNIST.DataStructures;

string BaseDatasetsRelativePath = @"C:\Users\firat\source\repos\MLDersleri\MulticlassClassificationMNIST\Data";
string TrianDataRealtivePath = $"{BaseDatasetsRelativePath}/optdigits-train.csv";
string TestDataRealtivePath = $"{BaseDatasetsRelativePath}/optdigits-val.csv";
string TrainDataPath = GetAbsolutePath(TrianDataRealtivePath);
string TestDataPath = GetAbsolutePath(TestDataRealtivePath);
string BaseModelsRelativePath = @"C:\Users\firat\source\repos\MLDersleri\MulticlassClassificationMNIST\MLModels";
string ModelRelativePath = $"{BaseModelsRelativePath}/Model.zip";
string ModelPath = GetAbsolutePath(ModelRelativePath);
//Veri ve model dosyalarının yollarını tanımlar ve mutlak yollara dönüştürür.

MLContext mlContext = new MLContext();
//ML.NET bağlamını (MLContext) oluşturur.

Train(mlContext);
//Modeli eğitir.
TestSomePredictions(mlContext);
//bazı tahminleri test eder.
Console.WriteLine("Hit any key to finish the app");
Console.ReadKey();

void Train(MLContext mlContext)//Eğitim ve test verilerini CSV dosyalarından yükler. Veri işleme ve eğitim boru hattını tanımlar. Modeli eğitir ve test verileri ile değerlendirir. Eğitilen modeli dosyaya kaydeder.
{
    try
    {
        //// Eğitim verilerini yükleme
        IDataView trainData = mlContext.Data.LoadFromTextFile(path: TrainDataPath, columns: new[]
        {//Eğitim verilerini LoadFromTextFile metodu ile TrainDataPath dosya yolundan yüklüyor. columns parametresi ile verinin sütunları tanımlanıyor:
            new TextLoader.Column(nameof(InputData.PixelValues), DataKind.Single, 0, 63),
            new TextLoader.Column("Number", DataKind.Single, 64)
        }, hasHeader: false, separatorChar: ',');
        //PixelValues sütunu, 0'dan 63'e kadar float türünde değerler içerir.
        //Number sütunu, 64. sütundaki float türünde değeri içerir.
        //hasHeader: false ile dosyanın başlık satırı olmadığı belirtiliyor.
        //separatorChar: ',' ile sütunlar arasındaki ayraç karakterinin virgül olduğu belirtiliyor.

        //Test verilerini yükleme
        IDataView testData = mlContext.Data.LoadFromTextFile(path: TestDataPath, columns: new[]
        {//Test verilerini aynı şekilde LoadFromTextFile metodu ile TestDataPath dosya yolundan yüklüyor ve aynı sütunları tanımlıyor.
            new TextLoader.Column(nameof(InputData.PixelValues), DataKind.Single, 0, 63),
            new TextLoader.Column("Number", DataKind.Single, 64)
        }, hasHeader: false, separatorChar: ',');

        //Veri işleme pipeline'ını oluşturma;
        EstimatorChain<TransformerChain<ColumnConcatenatingTransformer>> dataProcessPipeline =
            mlContext.Transforms.Conversion
            .MapValueToKey("Label", "Number", keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue)//Number sütunu etiket sütununa (Label) dönüştürülüyor.
            .Append(mlContext.Transforms//
            .Concatenate("Features", nameof(InputData.PixelValues))//PixelValues sütunları tek bir Features sütunu haline getiriliyor.
            .AppendCacheCheckpoint(mlContext));//İşlem durumu kontrol noktası ekleniyor.

        //Modeli eğitici (trainer) tanımlama: SdcaMaximumEntropy eğiticisi kullanılarak, Label ve Features sütunları belirtiliyor.
        SdcaMaximumEntropyMulticlassTrainer trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features");

        //Eğitim pipeline'ını oluşturma: Eğitim pipeline'ı oluşturuluyor. Veri işleme pipeline'ı ve eğitici birbirine ekleniyor.
        EstimatorChain<KeyToValueMappingTransformer> trainingPipeline = dataProcessPipeline
            .Append(trainer)
            .Append(mlContext.Transforms.Conversion
            .MapKeyToValue("Number", "Label"));//MapKeyToValue ile etiket sütunu Label, Number sütununa geri dönüştürülüyor.

        //Modelin eğitildiğini belirten mesaj ve Modeli eğitme;
        Console.WriteLine("=============== Training the model ===============");
        ITransformer trainedModel = trainingPipeline.Fit(trainData);

        //Modelin doğruluğunu test verisiyle değerlendirme;
        Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
        IDataView predictions = trainedModel.Transform(testData);//Eğitilmiş model kullanılarak test verileri üzerinden tahminler yapılıyor.
        MulticlassClassificationMetrics metrics = mlContext.MulticlassClassification.Evaluate(data: predictions, labelColumnName: "Number", scoreColumnName: "Score");
        //Modelin performansı Evaluate metodu ile değerlendirilerek çok sınıflı sınıflandırma metrikleri (MulticlassClassificationMetrics) hesaplanıyor.

        //Değerlendirme sonuçlarını konsola yazdırma;
        ConsoleHelper.PrintMultiClassClassificationMetrics(trainer.ToString(), metrics);

        //Eğitilmiş modeli kaydetme;
        mlContext.Model.Save(trainedModel, trainData.Schema, ModelPath);

        //Modelin kaydedildiğini belirten mesaj;
        Console.WriteLine("The model is saved to {0}", ModelPath);
    }
    catch (Exception ex)
    {
        //Hata yakalanırsa hatayı konsola yazdırma;
        Console.WriteLine(ex.ToString());
    }
}

void TestSomePredictions(MLContext mlContext)
{
    //Modeli yükleme;
    ITransformer trainedModel = mlContext.Model.Load(ModelPath, out var modelInputSchema);
    //Load metodu kullanılarak eğitimli model belirtilen dosya yolundan (ModelPath) yükleniyor. modelInputSchema değişkeni, modelin giriş şemasını içeriyor.

    //Tahmin motorunu oluşturma;
    PredictionEngine<InputData, OutPutData> predEngine = mlContext.Model.CreatePredictionEngine<InputData, OutPutData>(trainedModel);
    //CreatePredictionEngine metodu kullanılarak bir tahmin motoru (PredictionEngine) oluşturuluyor. Bu motor, InputData tipindeki girişleri alıp OutPutData tipindeki tahminleri döndürecek şekilde ayarlanıyor.

    //İlk örnek için tahmin yapma;
    OutPutData resultprediction1 = predEngine.Predict(SampleMNISTData.MNIST1);
    //Predict metodu kullanılarak ilk örnek veri (MNIST1) için tahmin yapılıyor ve sonuç resultprediction1 değişkenine atanıyor.

    //İlk örnek için tahmin sonuçlarını yazdırma;
    Console.WriteLine($"Actual: 1     Predicted probability:       zero:  {resultprediction1.Score[0]:0.####}");
    Console.WriteLine($"                                           One :  {resultprediction1.Score[1]:0.####}");
    Console.WriteLine($"                                           two:   {resultprediction1.Score[2]:0.####}");
    Console.WriteLine($"                                           three: {resultprediction1.Score[3]:0.####}");
    Console.WriteLine($"                                           four:  {resultprediction1.Score[4]:0.####}");
    Console.WriteLine($"                                           five:  {resultprediction1.Score[5]:0.####}");
    Console.WriteLine($"                                           six:   {resultprediction1.Score[6]:0.####}");
    Console.WriteLine($"                                           seven: {resultprediction1.Score[7]:0.####}");
    Console.WriteLine($"                                           eight: {resultprediction1.Score[8]:0.####}");
    Console.WriteLine($"                                           nine:  {resultprediction1.Score[9]:0.####}");
    Console.WriteLine();

    //İkinci örnek için tahmin yapma;
    OutPutData resultprediction2 = predEngine.Predict(SampleMNISTData.MNIST2);

    //İkinci örnek için tahmin sonuçlarını yazdırma;
    Console.WriteLine($"Actual: 7     Predicted probability:       zero:  {resultprediction2.Score[0]:0.####}");
    Console.WriteLine($"                                           One :  {resultprediction2.Score[1]:0.####}");
    Console.WriteLine($"                                           two:   {resultprediction2.Score[2]:0.####}");
    Console.WriteLine($"                                           three: {resultprediction2.Score[3]:0.####}");
    Console.WriteLine($"                                           four:  {resultprediction2.Score[4]:0.####}");
    Console.WriteLine($"                                           five:  {resultprediction2.Score[5]:0.####}");
    Console.WriteLine($"                                           six:   {resultprediction2.Score[6]:0.####}");
    Console.WriteLine($"                                           seven: {resultprediction2.Score[7]:0.####}");
    Console.WriteLine($"                                           eight: {resultprediction2.Score[8]:0.####}");
    Console.WriteLine($"                                           nine:  {resultprediction2.Score[9]:0.####}");
    Console.WriteLine();

    //Üçüncü örnek için tahmin yapma;
    OutPutData resultprediction3 = predEngine.Predict(SampleMNISTData.MNIST3);

    //Üçüncü örnek için tahmin sonuçlarını yazdırma;
    Console.WriteLine($"Actual: 9     Predicted probability:       zero:  {resultprediction3.Score[0]:0.####}");
    Console.WriteLine($"                                           One :  {resultprediction3.Score[1]:0.####}");
    Console.WriteLine($"                                           two:   {resultprediction3.Score[2]:0.####}");
    Console.WriteLine($"                                           three: {resultprediction3.Score[3]:0.####}");
    Console.WriteLine($"                                           four:  {resultprediction3.Score[4]:0.####}");
    Console.WriteLine($"                                           five:  {resultprediction3.Score[5]:0.####}");
    Console.WriteLine($"                                           six:   {resultprediction3.Score[6]:0.####}");
    Console.WriteLine($"                                           seven: {resultprediction3.Score[7]:0.####}");
    Console.WriteLine($"                                           eight: {resultprediction3.Score[8]:0.####}");
    Console.WriteLine($"                                           nine:  {resultprediction3.Score[9]:0.####}");
    Console.WriteLine();
}

string GetAbsolutePath(string relativePath)//Verilen göreli dosya yolunu mutlak dosya yoluna dönüştürür.
{
    FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
    string assemblyFolderPath = _dataRoot.Directory.FullName;
    string fullPath = Path.Combine(assemblyFolderPath, relativePath);
    return fullPath;
}
