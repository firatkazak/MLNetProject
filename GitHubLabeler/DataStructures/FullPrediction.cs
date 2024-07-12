namespace GitHubLabeler.DataStructures;
public class FullPrediction
{
    public string PredictedLabel;//Tahmin edilen etiket.
    public float Score;//Tahminin güven puanı.
    public int OriginalSchemaIndex;//Tahmin edilen etiketin indeksi.

    public FullPrediction(string predictedLabel, float score, int originalSchemaIndex)
    {
        PredictedLabel = predictedLabel;
        Score = score;
        OriginalSchemaIndex = originalSchemaIndex;
    }
}
//FullPrediction sınıfı, bir tahmin sonucunu temsil eder.