using Common;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using SpamDetection.MLDataStructures;

string AppPath = Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
//Uygulamanın çalıştığı dizini alır.
string DataDirectoryPath = Path.Combine(AppPath, @"C:\Users\firat\source\repos\MLDersleri\SpamDetection\Data");
//Veri dosyalarının bulunduğu dizini belirler.
string TrainDataPath = Path.Combine(AppPath, @"C:\Users\firat\source\repos\MLDersleri\SpamDetection\Data", "SMSSpamCollection");
//Eğitim veri dosyasının tam yolunu belirler.
MLContext mlContext = new MLContext();
//ML.NET'in merkezi sınıfıdır. Veri işlemlerini, model eğitimini ve değerlendirmesini sağlar.
IDataView data = mlContext.Data.LoadFromTextFile<SpamInput>(path: TrainDataPath, hasHeader: true, separatorChar: '\t');
//Eğitim veri setini yükler. SpamInput sınıfındaki öznitelikler, veri setindeki sütunlarla eşleştirilir.
EstimatorChain<ColumnCopyingTransformer> dataProcessPipeline = mlContext.Transforms.Conversion
    .MapValueToKey("Label", "Label")//Label sütununu anahtar değerlerine (numerik değerlere) dönüştürür.
    .Append(mlContext.Transforms.Text//
    .FeaturizeText("FeaturesText", new Microsoft.ML.Transforms.Text.TextFeaturizingEstimator.Options//Message sütununu özellikler haline getirir. Kelime ve karakter n-gramları kullanılır.
    {
        WordFeatureExtractor = new Microsoft.ML.Transforms.Text.WordBagEstimator.Options { NgramLength = 2, UseAllLengths = true },//İkili(2) n - gramlar oluşturur.
        CharFeatureExtractor = new Microsoft.ML.Transforms.Text.WordBagEstimator.Options { NgramLength = 3, UseAllLengths = false },//Üçlü(3) n - gramlar oluşturur.
        Norm = Microsoft.ML.Transforms.Text.TextFeaturizingEstimator.NormFunction.L2,//
    }, "Message"))
    .Append(mlContext.Transforms//L2 normalizasyonu uygular.
    .CopyColumns("Features", "FeaturesText"))//FeaturesText sütununu Features sütununa kopyalar.
    .AppendCacheCheckpoint(mlContext);//Veri işleme hattının performansını artırmak için önbellek noktası ekler.

EstimatorChain<KeyToValueMappingTransformer> trainer = mlContext.MulticlassClassification.Trainers
    .OneVersusAll(mlContext.BinaryClassification.Trainers//İkili sınıflandırıcıları (Averaged Perceptron) kullanarak çok sınıflı sınıflandırma modeli oluşturur.
    .AveragedPerceptron(labelColumnName: "Label", numberOfIterations: 10, featureColumnName: "Features"), labelColumnName: "Label")//Perceptron algoritmasını kullanır. Label sütununu hedefler ve Features sütununu kullanır.
    .Append(mlContext.Transforms.Conversion//
    .MapKeyToValue("PredictedLabel", "PredictedLabel"));//Modelin tahmin ettiği anahtar değerlerini etiketlere dönüştürür.

EstimatorChain<TransformerChain<KeyToValueMappingTransformer>> trainingPipeLine = dataProcessPipeline.Append(trainer);
//Veri işleme hattı ile eğitici hattı birleştirir.
Console.WriteLine("=============== Cross-validating to get model's accuracy metrics ===============");
//Konsola Cross-validating to get model's accuracy metrics yazdıracağız.
IReadOnlyList<TrainCatalogBase.CrossValidationResult<MulticlassClassificationMetrics>> crossValidationResults =
    mlContext.MulticlassClassification.CrossValidate(data: data, estimator: trainingPipeLine, numberOfFolds: 5);
//5 katlı çapraz doğrulama ile modelin performansını değerlendirir.
ConsoleHelper.PrintMulticlassClassificationFoldsAverageMetrics(trainer.ToString(), crossValidationResults);
//Çapraz doğrulama sonuçlarını yazdırır.
TransformerChain<TransformerChain<KeyToValueMappingTransformer>> model = trainingPipeLine.Fit(data);
//Modeli eğitim verileri ile eğitir ve eğitilmiş modeli döner.
PredictionEngine<SpamInput, SpamPrediction> predictor = mlContext.Model.CreatePredictionEngine<SpamInput, SpamPrediction>(model);
//Eğitilmiş modeli kullanarak tahmin motoru oluşturur.
Console.WriteLine("=============== Predictions for below data===============");
ClassifyMessage(predictor, "That's a great idea. It should work.");
ClassifyMessage(predictor, "free medicine winner! congratulations");
ClassifyMessage(predictor, "Yes we should meet over the weekend!");
ClassifyMessage(predictor, "you win pills and free entry vouchers");
//Verilen mesajlar için tahmin yapar ve sonucu yazdırır.
Console.WriteLine("=============== End of process, hit any key to finish =============== ");
Console.ReadLine();
//Kullanıcının giriş yapmasını bekler (programın kapanmasını önler).
void ClassifyMessage(PredictionEngine<SpamInput, SpamPrediction> predictor, string message)
{
    SpamInput input = new SpamInput { Message = message };//Yeni bir SpamInput nesnesi oluşturur ve Message özelliğine verilen mesajı atar.
    SpamPrediction prediction = predictor.Predict(input);//Tahmin motorunu kullanarak mesajın spam olup olmadığını tahmin eder.
    Console.WriteLine("The message '{0}' is {1}", input.Message, prediction.isSpam == "spam" ? "spam" : "not spam");//Tahmin sonucunu yazdırır.
}
//Özet: Bu proje, bir SMS mesajının spam olup olmadığını tespit etmek için ikili sınıflandırma kullanır.
//SpamInput: Veri setindeki her bir kaydı tanımlar.
//SpamPrediction: Modelin tahmin sonuçlarını tutar.
//Program.cs: Veri yükleme, işleme, model eğitimi, çapraz doğrulama ve tahmin işlemlerini gerçekleştirir.