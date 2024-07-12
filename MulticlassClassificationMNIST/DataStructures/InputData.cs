using Microsoft.ML.Data;

namespace MulticlassClassificationMNIST.DataStructures;
class InputData
{
    [ColumnName("PixelValues")]
    [VectorType(64)]
    public float[] PixelValues;//8x8 boyutundaki bir el yazısı rakamının 64 adet piksel değerini içeren bir dizi.

    [LoadColumn(64)]
    public float Number;//Bu el yazısı rakamının gerçek değeri (etiketi), yani hangi rakam olduğunu belirtir.
}
//Bu class, MNIST veri setindeki bir el yazısı rakamının piksel değerlerini ve bu rakamın gerçek değerini (etiketini) saklar.