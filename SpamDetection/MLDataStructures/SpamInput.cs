using Microsoft.ML.Data;

namespace SpamDetection.MLDataStructures;
public class SpamInput
{
    [LoadColumn(0)]//Veri setindeki 0. sütun Label özelliğine yüklenir.
    public string Label { get; set; }//Verinin spam olup olmadığını belirten etiket (spam veya ham).
    [LoadColumn(1)]//Veri setindeki 1. sütun Message özelliğine yüklenir.
    public string Message { get; set; }//Spam tespit edilmesi gereken SMS mesajı.
}
