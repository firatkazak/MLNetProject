using Microsoft.ML.Data;

namespace ClusteringIris.DataStructures;

//Bu sınıf, modelin tahmin sonuçlarını temsil eder. SelectedClusterId alanı, iris çiçeğinin hangi kümeye atandığını gösterirken, Distance alanı, bu atamanın doğruluğunu değerlendirmeye yardımcı olabilir.
public class IrisPrediction
{
    [ColumnName("PredictedLabel")]
    public uint SelectedClusterId;//Model tarafından tahmin edilen, çiçeğin hangi kümeye ait olduğunu belirten ID (kimlik) numarası.

    [ColumnName("Score")]
    public float[] Distance;//Her bir veri noktasının (çiceğin) merkezden olan uzaklıkları. Bu dizi, tahmin edilen kümeye olan mesafeyi ve olasılıkla diğer kümelere olan mesafeyi içerir.
}
