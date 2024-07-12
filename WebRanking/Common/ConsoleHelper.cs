using Microsoft.ML.Data;
using Microsoft.ML;
using WebRanking.DataStructures;
using System.Reflection;

namespace WebRanking.Common;

//ConsoleHelper sınıfı, makine öğrenimi modelinizin performansını değerlendirip sonuçları konsolda görüntülemenize yardımcı olur. Özellikle, DCG ve NDCG gibi sıralama metriklerini hesaplayıp görüntüler ve tahmin edilen skorları yazdırır;
public class ConsoleHelper
{
    public static void EvaluateMetrics(MLContext mlContext, IDataView predictions)
    {
        //metrics: ML.NET bağlamını (MLContext) kullanarak sıralama metriklerini (RankingMetrics) değerlendirir.
        RankingMetrics metrics = mlContext.Ranking.Evaluate(predictions);

        //Discounted Cumulative Gain: Sıralanmış sonuçların kalitesini ölçen bir metriktir. Sıralamanın ilk pozisyonlarındaki yüksek kaliteli sonuçlara daha fazla ağırlık verir;
        Console.WriteLine($"DCG: {string.Join(", ", metrics.DiscountedCumulativeGains.Select((d, i) => $"@{i + 1}:{d:F4}").ToArray())}");

        //Normalized Discounted Cumulative Gain: DCG'nin normalize edilmiş versiyonudur, sonuçların daha kolay karşılaştırılmasını sağlar.
        Console.WriteLine($"NDCG: {string.Join(", ", metrics.NormalizedDiscountedCumulativeGains.Select((d, i) => $"@{i + 1}:{d:F4}").ToArray())}\n");
    }

    //Bu metot, belirtilen truncation level'a göre değerlendirme metriklerini hesaplar. Kodun çoğu, RankingEvaluator sınıfının dinamik olarak oluşturulmasını içerir;
    public static void EvaluateMetrics(MLContext mlContext, IDataView predictions, int truncationLevel)
    {//DCG ve NDCG hesaplamalarında kaç sonuç göz önüne alınacağını belirler (1-10 arasında).
        if (truncationLevel < 1 || truncationLevel > 10)
        {
            throw new InvalidOperationException("Currently metrics are only supported for 1 to 10 truncation levels.");
        }

        //mlAssembly: ML.NET'in TextLoader sınıfının bulunduğu assembly'yi yükler;
        Assembly mlAssembly = typeof(TextLoader).Assembly;
        //rankEvalType: RankingEvaluator türünü elde eder;
        TypeInfo rankEvalType = mlAssembly.DefinedTypes.Where(t => t.Name.Contains("RankingEvaluator")).First();
        //rankEvalType (RankingEvaluator tipi) içinde tanımlı Arguments adlı alt sınıfın tipini elde eder;
        Type evalArgsType = rankEvalType.GetNestedType("Arguments");
        //evalArgs: RankingEvaluator'ün Arguments alt sınıfının bir örneğini oluşturur;
        object evalArgs = Activator.CreateInstance(rankEvalType.GetNestedType("Arguments"));
        //Arguments alt sınıfında tanımlı DcgTruncationLevel alanının bilgilerini alır;
        FieldInfo dcgLevel = evalArgsType.GetField("DcgTruncationLevel");
        //evalArgs nesnesindeki DcgTruncationLevel alanına truncationLevel değerini atar;
        dcgLevel.SetValue(evalArgs, truncationLevel);
        //RankingEvaluator sınıfının ilk (ve muhtemelen tek) kurucusunu alır;
        ConstructorInfo ctor = rankEvalType.GetConstructors().First();
        //evaluator: RankingEvaluator sınıfının bir örneğini oluşturur.
        object evaluator = ctor.Invoke(new object[] { mlContext, evalArgs });
        //evaluateMethod: RankingEvaluator'ün Evaluate metodunu çağırarak metrikleri hesaplar.
        MethodInfo evaluateMethod = rankEvalType.GetMethod("Evaluate");
        //evaluateMethod metodunu çağırarak sıralama metriklerini hesaplar ve sonucu RankingMetrics tipine dönüştürür.
        RankingMetrics metrics = (RankingMetrics)evaluateMethod.Invoke(evaluator, new object[] { predictions, "Label", "GroupId", "Score" });
        //Çıktı;
        Console.WriteLine($"DCG: {string.Join(", ", metrics.DiscountedCumulativeGains.Select((d, i) => $"@{i + 1}:{d:F4}").ToArray())}");
        //Çıktı;
        Console.WriteLine($"NDCG: {string.Join(", ", metrics.NormalizedDiscountedCumulativeGains.Select((d, i) => $"@{i + 1}:{d:F4}").ToArray())}\n");
    }

    //Bu metot, her bir tahminin GroupId ve Score değerlerini konsola yazdırır.
    public static void PrintScores(IEnumerable<SearchResultPrediction> predictions)
    {//predictions: Modelin tahmin ettiği sonuçlar (SearchResultPrediction tipinde).
        foreach (SearchResultPrediction prediction in predictions)
        {
            Console.WriteLine($"GroupId: {prediction.GroupId}, Score: {prediction.Score}");
        }
    }
}
