using Microsoft.ML.Data;

namespace SpikeDetection.DataStructures;
public class ProductSalesData
{
    [LoadColumn(0)]
    public string Month;

    [LoadColumn(1)]
    public float numSales;
}
//ProductSalesData sınıfı ise modelin eğitim ve test verilerini temsil ediyor. CSV dosyasındaki sütunların hangi sınıf özelliklerine yükleneceğini belirten [LoadColumn] özniteliği ile tanımlanmış.