using Microsoft.ML.Data;

namespace ObjectDetection.DataStructures;
public class ImageNetData
{
    [LoadColumn(0)]
    public string ImagePath;//Resmin dosya yolunu tutar (LoadColumn(0) ile belirtilir).

    [LoadColumn(1)]
    public string Label;//Resmin etiketini tutar (LoadColumn(1) ile belirtilir).

    //ReadFromFile: Belirtilen bir klasördeki resim dosyalarını okuyarak ImageNetData nesneleri oluşturur.
    public static IEnumerable<ImageNetData> ReadFromFile(string imageFolder)
    {
        return Directory
            .GetFiles(imageFolder)
            .Where(filePath => Path.GetExtension(filePath) != ".md")
            .Select(filePath => new ImageNetData { ImagePath = filePath, Label = Path.GetFileName(filePath) });
    }
}
//ImageNetData: ML.NET modeline veri sağlamak için kullanılır.