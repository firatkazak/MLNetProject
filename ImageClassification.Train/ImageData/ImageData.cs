using Microsoft.ML.Data;

namespace ImageClassification.Train.ImageData;

public class ImageData
{
    [LoadColumn(0)]
    public string ImagePath;//Resmin dosya yolunu temsil eden bir string.

    [LoadColumn(1)]//LoadColumn özniteliği, bu verilerin CSV veya benzeri bir dosyadan yüklenmesi gerektiğini belirtir.
    public string Label;//Resmin etiketini (örneğin çiçek türü) temsil eden bir string.
}
