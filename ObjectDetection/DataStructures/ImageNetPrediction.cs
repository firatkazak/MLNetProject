using Microsoft.ML.Data;

namespace ObjectDetection.DataStructures;
public class ImageNetPrediction
{
    [ColumnName("grid")]//Modelin tahmin ettiği etiketlerin sütun adını belirtir.
    public float[] PredictedLabels;//Tahmin edilen etiketlerin bir dizisini içerir.
}
//ImageNetPrediction: ML.NET modelinizin tahmin ettiği etiketleri tutmak için kullanılır.