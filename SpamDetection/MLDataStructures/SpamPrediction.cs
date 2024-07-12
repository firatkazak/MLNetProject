using Microsoft.ML.Data;

namespace SpamDetection.MLDataStructures;
public class SpamPrediction
{
    [ColumnName("PredictedLabel")]//Model tarafından üretilen tahminin isSpam özelliğine yüklenmesini sağlar.
    public string isSpam { get; set; }//Modelin tahmin ettiği etiket (spam veya ham).
}
