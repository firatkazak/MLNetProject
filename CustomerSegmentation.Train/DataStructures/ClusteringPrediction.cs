using Microsoft.ML.Data;

namespace CustomerSegmentation.Train.DataStructures;
public class ClusteringPrediction
{
    [ColumnName("PredictedLabel")]
    public uint SelectedClusterId;//Tahmin edilen kümenin ID'si.
    [ColumnName("Score")]
    public float[] Distance;//Her küme için mesafeler.
    [ColumnName("PCAFeatures")]
    public float[] Location;//PCA (Principal Component Analysis) ile elde edilen özelliklerin konumu.
    [ColumnName("LastName")]
    public string LastName;//Müşterinin soyadı.
}
//ClusteringPrediction sınıfı, bir kümeleme modeli tarafından yapılan tahminlerin sonuçlarını saklar.