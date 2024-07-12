using Microsoft.ML.Data;

namespace HeartDiseaseDetection.DataStructures;
//Bu sınıf, kalp hastalığı veri setindeki her bir örneği temsil eder.
//Property'lerin her biri, veri setindeki sütunlara karşılık gelir.
//LoadColumn Attribute'u ile sıfırdan başlayarak belirtilen sütun indeksini belirtir.
//Bu sınıf, eğitim ve test veri setlerindeki her bir özniteliği (yaş, cinsiyet, vb.) ve sonucu (etiket) temsil eder.

public class HeartData
{
    [LoadColumn(0)]
    public float Age { get; set; }
    [LoadColumn(1)]
    public float Sex { get; set; }
    [LoadColumn(2)]
    public float Cp { get; set; }
    [LoadColumn(3)]
    public float TrestBps { get; set; }
    [LoadColumn(4)]
    public float Chol { get; set; }
    [LoadColumn(5)]
    public float Fbs { get; set; }
    [LoadColumn(6)]
    public float RestEcg { get; set; }
    [LoadColumn(7)]
    public float Thalac { get; set; }
    [LoadColumn(8)]
    public float Exang { get; set; }
    [LoadColumn(9)]
    public float OldPeak { get; set; }
    [LoadColumn(10)]
    public float Slope { get; set; }
    [LoadColumn(11)]
    public float Ca { get; set; }
    [LoadColumn(12)]
    public float Thal { get; set; }
    [LoadColumn(13)]
    public bool Label { get; set; }
}
