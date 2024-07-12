using ImageClassification.Train.ImageData;
using ImageClassification.Train.Model;
using Microsoft.ML;

namespace ImageClassification.Predict.Model;

//ModelScorer sınıfı, verilen bir model dosyasını yükler ve belirtilen klasördeki resimleri sınıflandırır.
public class ModelScorer
{
    private readonly string imagesFolder;//imagesFolder: Sınıflandırılacak resimlerin bulunduğu klasörün yolu.
    private readonly string modelLocation;//modelLocation: Eğitilmiş model dosyasının yolu.
    private readonly MLContext mlContext;//mlContext: ML.NET işlemleri için kullanılan MLContext nesnesi.

    public ModelScorer(string imagesFolder, string modelLocation)
    {
        this.imagesFolder = imagesFolder;
        this.modelLocation = modelLocation;
        mlContext = new MLContext(seed: 1);
    }

    //ClassifyImages metodu, modeli yükler, resimleri tahmin eder ve sonuçları yazdırır.
    public void ClassifyImages()
    {
        //Konsola başlık yazar;
        ConsoleHelpers.ConsoleWriteHeader("Loading model");
        Console.WriteLine("");
        Console.WriteLine($"Model loaded: {modelLocation}");

        //Modeli belirtilen konumdan yükler ve giriş şemasını alır.
        ITransformer loadedModel = mlContext.Model.Load(modelLocation, out DataViewSchema modelInputSchema);

        //Tahmin motoru oluşturur;
        PredictionEngine<ImageData, ImagePrediction> predictionEngine = mlContext.Model.CreatePredictionEngine<ImageData, ImagePrediction>(loadedModel);

        //Belirtilen klasörden resimleri yükler;
        IEnumerable<ImageData> imagesToPredict = LoadImagesFromDirectory(imagesFolder, true);

        ConsoleHelpers.ConsoleWriteHeader("Predicting classifications...");

        //imagesToPredict koleksiyonundan ilk resmi alır ve bu resmi tahmin etmek için ImageData nesnesi olarak hazırlar.
        ImageData imageToPredict = new ImageData
        {
            ImagePath = imagesToPredict.First().ImagePath
        };
        //İlk resmi tahmin eder ve sonucu yazdırır.
        ImagePrediction prediction = predictionEngine.Predict(imageToPredict);

        Console.WriteLine("");
        Console.WriteLine($"ImageFile : [{Path.GetFileName(imageToPredict.ImagePath)}], " +
                          $"Scores : [{string.Join(",", prediction.Score)}], " +
                          $"Predicted Label : {prediction.PredictedLabelValue}");

        Console.WriteLine("");
        Console.WriteLine("Predicting several images...");

        foreach (ImageData currentImageToPredict in imagesToPredict)
        {
            ImagePrediction currentPrediction = predictionEngine.Predict(currentImageToPredict);
            Console.WriteLine("");
            Console.WriteLine($"ImageFile : [{Path.GetFileName(currentImageToPredict.ImagePath)}], " +
                              $"Scores : [{string.Join(",", currentPrediction.Score)}], " +
                              $"Predicted Label : {currentPrediction.PredictedLabelValue}");
        }
    }

    //LoadImagesFromDirectory metodu, resimleri belirtilen klasörden yükler ve ImageData nesneleri oluşturur.
    //string folder: Resimlerin bulunduğu dizinin yolu.
    //bool useFolderNameasLabel: Etiketlerin belirlenmesinde klasör adlarının kullanılıp kullanılmayacağını belirten opsiyonel bir parametre. Varsayılan değeri true.
    public static IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameasLabel = true)
    {
        //Directory.GetFiles metodu, belirtilen dizindeki (ve alt dizinlerdeki) tüm dosyaların yollarını alır. * joker karakteri tüm dosyaları seçer.
        string[] files = Directory.GetFiles(folder, "*", searchOption: SearchOption.AllDirectories);

        //Bulunan dosyaların her biri için döngü başlatılır;
        foreach (string file in files)
        {
            //Dosyanın uzantısı .jpg veya .png değilse, döngünün o adımını atlar (continue).
            if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
                continue;

            //Bu satır, dosyanın tam yolundan sadece dosya adını alır. Örneğin, dosya yolu "C:\Users\firat\images\flower.jpg" ise, Path.GetFileName(file) ifadesi "flower.jpg" değerini döndürür.
            string label = Path.GetFileName(file);

            //useFolderNameasLabel parametresi true ise, etiket olarak dosyanın bulunduğu klasörün adı kullanılır.
            if (useFolderNameasLabel)
                label = Directory.GetParent(file).Name;
            else//useFolderNameasLabel false ise, etiket olarak dosya adındaki harfler alınır. İlk harf olmayan karakter bulunduğunda etiket bu noktada kesilir.
            {
                for (int index = 0; index < label.Length; index++)
                {
                    if (!char.IsLetter(label[index]))
                    {
                        label = label.Substring(0, index);
                        break;
                    }
                }
            }
            //Yeni bir ImageData nesnesi oluşturulur ve ImagePath ile Label özellikleri atanır. yield return ifadesi, LoadImagesFromDirectory metodunun bir IEnumerable<ImageData> olarak resimleri adım adım döndürmesini sağlar.
            yield return new ImageData()
            {
                ImagePath = file,
                Label = label
            };

        }
    }
}
