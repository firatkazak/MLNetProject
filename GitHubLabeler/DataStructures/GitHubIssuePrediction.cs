using Microsoft.ML.Data;

namespace GitHubLabeler.DataStructures;
internal class GitHubIssuePrediction
{
    [ColumnName("PredictedLabel")]
    public string Area;//Tahmin edilen etiket.
    public float[] Score;//Tahmin edilen etiketlerin puanları.
}
//GitHubIssuePrediction sınıfı, bir tahmin sonucunu temsil eder.