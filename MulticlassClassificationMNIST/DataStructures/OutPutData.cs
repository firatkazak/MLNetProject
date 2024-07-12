using Microsoft.ML.Data;

namespace MulticlassClassificationMNIST.DataStructures;
class OutPutData
{
    [ColumnName("Score")]
    public float[] Score;//Modelin tahmin ettiği olasılıkları içeren bir dizi. Her bir eleman, modelin belirli bir rakam olma olasılığını gösterir.
}
//Bu class, modelin tahmin ettiği sonuçları saklar.
