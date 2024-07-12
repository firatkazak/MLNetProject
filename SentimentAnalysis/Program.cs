using Common;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;
using SentimentAnalysis.DataStructures;
using static Microsoft.ML.DataOperationsCatalog;

string DataPath = @"C:\Users\firat\source\repos\MLDersleri\SentimentAnalysis\Data\wikiDetoxAnnotated40kRows.tsv";
//Verimizin olduğu path.
string ModelPath = @"C:\Users\firat\source\repos\MLDersleri\SentimentAnalysis\Model\SentimentModel.zip";
//Çıktının oluşturulacağı path.
MLContext mlContext = new MLContext(seed: 1);
//ML.NET bağlamını (context) oluşturur. Rastgele işlemler için bir başlangıç değeri (seed) belirler.
IDataView dataView = mlContext.Data.LoadFromTextFile<SentimentIssue>(DataPath, hasHeader: true);
//DataPath'teki TSV dosyasını SentimentIssue sınıfına göre yükler.
TrainTestData trainTestSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
//Veriyi eğitim (%80) ve test (%20) setlerine böler.
IDataView trainingData = trainTestSplit.TrainSet;
//Eğitim veri setlerini belirtir.
IDataView testData = trainTestSplit.TestSet;
//Test veri setlerini belirtir.
TextFeaturizingEstimator dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(SentimentIssue.Text));
//Metin verisini sayısal özelliklere dönüştürür ve bu özellikleri Features sütununa yazar.
SdcaLogisticRegressionBinaryTrainer trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features");
//SDCA Lojistik Regresyon algoritmasını kullanarak ikili sınıflandırma modeli oluşturur.
EstimatorChain<BinaryPredictionTransformer<CalibratedModelParametersBase<LinearBinaryModelParameters, PlattCalibrator>>> trainingPipeline = dataProcessPipeline.Append(trainer);
//Veri işleme ve model eğitim aşamalarını birleştirir.
ITransformer trainedModel = trainingPipeline.Fit(trainingData);
//Eğitim verisiyle modeli eğitir.
IDataView predictions = trainedModel.Transform(testData);
//Test verisiyle modeli kullanarak tahminler yapar.
CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label", scoreColumnName: "Score");
//Modelin performansını değerlendirir.
ConsoleHelper.PrintBinaryClassificationMetrics(trainer.ToString(), metrics);
//Değerlendirme metriklerini konsola yazdırır.
mlContext.Model.Save(trainedModel, trainingData.Schema, ModelPath);
//Eğitilmiş modeli belirtilen yola kaydeder.
Console.WriteLine("The model is saved to {0}", ModelPath);
//Kaydedilen verinin bilgisini verir.
SentimentIssue sampleStatement = new SentimentIssue { Text = "I love this movie!" };
//*** 2 SATIR ALTTAKİ KODA BAK.
PredictionEngine<SentimentIssue, SentimentPrediction> predEngine = mlContext.Model.CreatePredictionEngine<SentimentIssue, SentimentPrediction>(trainedModel);
//Tek bir örnek üzerinde tahmin yapmak için tahmin motoru oluşturur.
SentimentPrediction resultprediction = predEngine.Predict(sampleStatement);
//Yeni bir metin (örneğin, "I love this movie!") üzerinde duygu analizi tahmini yapar. 2 SATIR ÜSTTEKİ KOD İLE BAĞLANTILI
Console.WriteLine($"=============== Single Prediction  ===============");
Console.WriteLine($"Text: {sampleStatement.Text} | Prediction: {(Convert.ToBoolean(resultprediction.Prediction) ? "Toxic" : "Non Toxic")} sentiment | Probability of being toxic: {resultprediction.Probability} ");
Console.WriteLine($"================End of Process.Hit any key to exit==================================");
Console.ReadLine();
//Tahmin sonucunu ve olasılığı konsola yazdırır ve programın bitmesini bekler.