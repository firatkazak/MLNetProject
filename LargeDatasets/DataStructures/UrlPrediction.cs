using Microsoft.ML.Data;

namespace LargeDatasets.DataStructures;
public class UrlPrediction
{
    [ColumnName("PredictedLabel")]
    public bool Prediction;

    public float Score;
}
