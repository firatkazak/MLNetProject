using Microsoft.ML.Data;

namespace BikeSharingDemand.DataStructures;
public class DemandPrediction
{
    [ColumnName("Score")]//ColumnName niteliğiyle ML.NET'e bu sütunun model çıktısını temsil ettiğini belirtiyoruz.
    public float PredictedCount;//PredictedCount: Bu özellik, tahmin edilen talep sayısını (ya da başka bir değişkeni) temsil eder.
}
