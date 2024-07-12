namespace CCFraudDetection.Common.DataModels;
public class TransactionFraudPrediction : IModelEntity
{
    public bool Label;//Bu özellik, işlemin sahte olup olmadığını belirtir (doğru/yanlış).
    public bool PredictedLabel;//Bu özellik, modelin tahmin ettiği etiketidir.
    public float Score;//Modelin verdiği güven skoru.
    public float Probability;//Tahminin olasılığı.

    public void PrintToConsole()
    {
        Console.WriteLine($"Predicted Label: {PredictedLabel}");
        Console.WriteLine($"Probability: {Probability}  ({Score})");
    }
}
//Amacı: modelin bir işlemi tahmin ederken ürettiği sonuçları temsil eder. Bu sınıf, modelin verdiği tahminlerin ve bu tahminlerin güvenilirlik skorlarını içerir.
//İşlevi:
//Etiket: Label özelliği, işlemin gerçek etiketini temsil eder(bu, modelin tahmin ederken kullanacağı doğru etikettir).
//Tahmin Edilen Etiket: PredictedLabel özelliği, modelin tahmin ettiği etiketi temsil eder.
//Skor: Score özelliği, modelin tahmininin güvenilirlik skorunu temsil eder.
//Olasılık: Probability özelliği, tahminin olasılığını temsil eder.
//Tahmin Görselleştirme: PrintToConsole() yöntemi, tahmin edilen etiketi, olasılığı ve skoru konsola yazdırır.