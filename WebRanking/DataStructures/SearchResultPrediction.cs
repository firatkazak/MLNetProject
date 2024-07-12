namespace WebRanking.DataStructures;

//Arama sonuçlarını tahmin eden bir modelin tahmin ettiği veriyi temsil eder. Bu sınıf, modelin tahmin sonuçlarını içeren özelliklere sahiptir.
public class SearchResultPrediction
{
    public uint GroupId { get; set; }//Tahmin edilen sonuçların gruplandırıldığı benzersiz kimliği temsil eder.
    public uint Label { get; set; }//Gerçek sınıf etiketini veya tahmin edilen sınıf etiketini temsil eder.
    public float Score { get; set; }//Modelin tahmin ettiği skoru temsil eder.
    public float[] Features { get; set; }//Modelin tahmin yaparken kullandığı özelliklerin bir dizisini temsil eder.
}
