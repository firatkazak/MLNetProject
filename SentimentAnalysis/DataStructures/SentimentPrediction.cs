using Microsoft.ML.Data;

namespace SentimentAnalysis.DataStructures;
public class SentimentPrediction
{
    [ColumnName("PredictedLabel")]//ColumnName niteliği, sütun adını varsayılan değeri olan alan adından değiştirmek için kullanılır.
    public bool Prediction { get; set; }
    public float Probability { get; set; }//ColumnName niteliğini belirtmemize gerek yok, çünkü "Probability" alan adı istediğimiz sütun adıdır.
    public float Score { get; set; }
}
//Amaç: Bu class, modelin tahmin sonuçlarını tutmak için kullanılır.Bu sınıf, modelin bir metin için yaptığı tahmini, bu tahmine olan güveni ve ham skorunu depolar.
//Prediction: Bu özellik, modelin tahmin ettiği duygu etiketini (pozitif veya negatif) belirtir.
//Probability: Bu özellik, tahminin olasılığını ifade eder. Modelin bu tahmine olan güven derecesini gösterir.
//Score: Bu özellik, modelin ham skorunu belirtir. Genellikle içsel model kararının bir ölçüsüdür ve olasılığa dönüştürülmeden önceki değerdir.
