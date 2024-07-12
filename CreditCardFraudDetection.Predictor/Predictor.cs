using CreditCardFraudDetection.Common.DataModels;
using Microsoft.ML;
using System.Data;

namespace CreditCardFraudDetection.Predictor;
public class Predictor
{
    private readonly string _modelfile;//Kaydedilmiş modelin dosya yolu.
    private readonly string _dasetFile;//Test verilerinin dosya yolu.

    public Predictor(string modelfile, string dasetFile)//Eğer herhangi bir parametre null ise, ArgumentNullException fırlatır.
    {
        _modelfile = modelfile ?? throw new ArgumentNullException(nameof(modelfile));
        _dasetFile = dasetFile ?? throw new ArgumentNullException(nameof(dasetFile));
    }

    public void RunMultiplePredictions(int numberOfPredictions)
    {
        //Yeni bir ML.NET bağlamı oluşturur;
        MLContext mlContext = new MLContext();
        //_dasetFile dosyasından verileri yükler ve IDataView formatına dönüştürür;
        IDataView inputDataForPredictions = mlContext.Data.LoadFromTextFile<TransactionObservation>(_dasetFile, separatorChar: ',', hasHeader: true);
        //Konsola modelden yapılacak tahminlerin başlığını yazdırır;
        Console.WriteLine($"Predictions from saved model:");
        //_modelfile dosyasından modeli yükler;
        ITransformer model = mlContext.Model.Load(_modelfile, out DataViewSchema inputSchema);
        //Yüklenen modelden bir tahmin motoru oluşturur;
        PredictionEngine<TransactionObservation, TransactionFraudPrediction> predictionEngine =
            mlContext.Model.CreatePredictionEngine<TransactionObservation, TransactionFraudPrediction>(model);
        //Konsola sahtekarlık olarak tahmin edilmesi gereken işlemlerin başlığını yazdırır;
        Console.WriteLine($"\n \n Test {numberOfPredictions} transactions, from the test datasource, that should be predicted as fraud (true):");
        //Veriyi TransactionObservation tipinde bir enumerable'a dönüştürür;
        mlContext.Data.CreateEnumerable<TransactionObservation>(inputDataForPredictions, reuseRowObject: false)
                    .Where(x => x.Label > 0)
                    .Take(numberOfPredictions)
                    .Select(testData => testData)
                    .ToList()
                    .ForEach(testData =>
                    {
                        Console.WriteLine($"--- Transaction ---");
                        testData.PrintToConsole();
                        //Tahmin motorunu kullanarak tahmin yapar ve tahmini konsola yazdırır;
                        predictionEngine.Predict(testData).PrintToConsole();
                        Console.WriteLine($"-------------------");
                    });
        //Konsola sahtekarlık olarak tahmin edilmemesi gereken işlemlerin başlığını yazdırır.
        Console.WriteLine($"\n \n Test {numberOfPredictions} transactions, from the test datasource, that should NOT be predicted as fraud (false):");
        //Veriyi TransactionObservation tipinde bir enumerable'a dönüştürür.
        mlContext.Data.CreateEnumerable<TransactionObservation>(inputDataForPredictions, reuseRowObject: false)
                   .Where(x => x.Label < 1)//Sahtekarlık olarak işaretlenmemiş (Label < 1) numberOfPredictions kadar işlemi alır ve listeye dönüştürür.
                   .Take(numberOfPredictions)
                   .ToList()
                   .ForEach(testData =>
                   {
                       Console.WriteLine($"--- Transaction ---");
                       testData.PrintToConsole();//İşlemi konsola yazdırır.

                       //Tahmin motorunu kullanarak tahmin yapar ve tahmini konsola yazdırır;
                       predictionEngine.Predict(testData).PrintToConsole();
                       Console.WriteLine($"-------------------");
                   });
    }
}
