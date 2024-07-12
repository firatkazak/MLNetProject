using Microsoft.ML.Data;

namespace ImageClassification.TensorFlow.ImageDataStructures;

public class ImageNetData//Veri okuma ve temsil etme işlemlerini yöneten sınıf.
{
    [LoadColumn(0)]
    public string ImagePath;//Resmin dosya yolunu temsil eder. LoadColumn niteliği, bu özelliğin veri dosyasındaki hangi sütundan yüklendiğini belirtir.

    [LoadColumn(1)]
    public string Label;//Resmin etiketini (label) temsil eder. Aynı şekilde LoadColumn niteliği ile veri dosyasındaki sütunu belirtir.

    //Bu metot, bir CSV dosyasından veri okur ve ImageNetData nesneleri koleksiyonu olarak döndürür. Her satırı okurken, resmin dosya yolunu ve etiketini belirleyerek yeni ImageNetData nesneleri oluşturur.
    public static IEnumerable<ImageNetData> ReadFromCsv(string file, string folder)
    {
        return File.ReadAllLines(file).Select(x => x.Split('\t')).Select(x => new ImageNetData { ImagePath = Path.Combine(folder, x[0]), Label = x[1] });
    }
}

//ImageNetDataProbability: Tahmin sonuçlarını saklamak ve işlemek için kullanılıyor;
public class ImageNetDataProbability : ImageNetData
{
    public string PredictedLabel;//TensorFlow modeli tarafından tahmin edilen etiketi (label) temsil eder.
    public float Probability { get; set; }//Bu tahminin olasılığını (probability) belirtir.
}
