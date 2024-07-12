using Microsoft.ML.Data;

namespace SentimentAnalysis.DataStructures;
public class SentimentIssue
{
    [LoadColumn(0)]//Veri setinin ilk sütunu Label'a denk gelecek şekilde ayarladık.
    public bool Label { get; set; }
    [LoadColumn(2)]//Veri setinin üçüncü sütunu Text'e denk gelecek şekilde ayarladık.
    public string Text { get; set; }
}
//Amaç:SentimentIssue sınıfı, veri setindeki her bir kaydı temsil eder.Bu sınıfın amacı, veri setindeki metin (text) ve etiket (label) bilgilerini tutmaktır.
//Label: Bu özellik, metnin taşıdığı duygu etiketini belirtir.Genellikle "pozitif" veya "negatif" olarak sınıflandırılır. Veri setinde duygu analizine tabi tutulacak metinlerin doğru ya da yanlış olarak etiketlendiğini gösterir.
//Text: Bu özellik, analiz edilecek olan metin verisini tutar. Metnin kendisini ifade eder.