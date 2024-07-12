namespace CreditCardFraudDetection.Common.DataModels;
public class TransactionFraudPrediction : IModelEntity
{
    public float Label;//Gerçek etiket (doğru olup olmadığı).
    public float Score;//Modelin verdiği tahmin skoru.
    public bool PredictedLabel;//Modelin tahmin ettiği etiket (doğru veya yanlış).

    //Tahmin edilen etiketi ve tahmin skorunu konsola yazdırır;
    public void PrintToConsole()
    {
        Console.WriteLine($"Predicted Label: {PredictedLabel}  (Score: {Score})");
    }
}
