using Microsoft.ML.Data;

namespace HeartDiseaseDetection.DataStructures;

//Bu sınıf, bir HeartData örneğinin tahminini temsil eder.
//ColumnName özniteliği, tahmin edilen etiketin adını belirtir ve diğer iki özellik, tahminin olasılığını ve skorunu tutar.

public class HeartPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Prediction;
    public float Probability;
    public float Score;
}
