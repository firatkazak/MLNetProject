using Microsoft.ML.Data;

namespace SpikeDetection.DataStructures;
public class ProductSalesPrediction
{
    [VectorType(3)]
    public double[] Prediction { get; set; }
}
//ProductSalesPrediction sınıfı, modelin tahmin ettiği değerleri tutmak için kullanılıyor. Bu örnekte, tahmin vektörü üç elemanlı bir dizi olarak tanımlanmış.