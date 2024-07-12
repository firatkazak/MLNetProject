using Common;
using ImageClassification.Train.ImageData;
using ImageClassification.Train.Model;

string assetsRelativePath = @"C:\Users\firat\source\repos\MLDersleri\ImageClassification.Train\assets";
string assetsPath = GetAbsolutePath(assetsRelativePath);
string inceptionPb = Path.Combine(assetsPath, "inputs", "tensorflow-pretrained-models", "inception-v3", "inception_v3_2016_08_28_frozen.pb");
string imageClassifierZip = Path.Combine(assetsPath, "outputs", "imageClassifier.zip");
string tagsTsv = Path.Combine(assetsPath, "inputs", "data", "tags.tsv");
string imagesDownloadFolderPath = Path.Combine(assetsPath, "inputs", "images");
string finalImagesFolderName = DownloadImageSet(imagesDownloadFolderPath);
string fullImagesetFolderPath = Path.Combine(imagesDownloadFolderPath, finalImagesFolderName);

Console.WriteLine($"Images folder: {fullImagesetFolderPath}");

//Resimleri belirtilen klasörden yükler ve her resim için ImageData nesneleri oluşturur.
IEnumerable<ImageData> allImages = LoadImagesFromDirectory(folder: fullImagesetFolderPath, useFolderNameasLabel: true);
try
{
    //ModelBuilder: TensorFlow modelini kullanarak ML.NET modelini eğitmek ve kaydetmek için bir yardımcı sınıf.
    ModelBuilder modelBuilder = new ModelBuilder(inceptionPb, imageClassifierZip);
    //BuildAndTrain: Resim verilerini kullanarak modeli eğitir.
    modelBuilder.BuildAndTrain(allImages);
}
catch (Exception ex)
{
    ConsoleHelpers.ConsoleWriteException(ex.ToString());
}

ConsoleHelpers.ConsolePressAnyKey();

//LoadImagesFromDirectory: Belirtilen klasördeki resimleri yükler ve her bir resim için ImageData nesneleri oluşturur.
//Klasördeki tüm dosyaları tarar, jpg ve png dosyalarını alır, etiket olarak dosya adını veya klasör adını kullanır.
IEnumerable<ImageData> LoadImagesFromDirectory(string folder, bool useFolderNameasLabel = true)
{
    string[] files = Directory.GetFiles(folder, "*", searchOption: SearchOption.AllDirectories);

    foreach (string file in files)
    {
        if ((Path.GetExtension(file) != ".jpg") && (Path.GetExtension(file) != ".png"))
            continue;

        string label = Path.GetFileName(file);
        if (useFolderNameasLabel)
            label = Directory.GetParent(file).Name;
        else
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
        yield return new ImageData()
        {
            ImagePath = file,
            Label = label
        };
    }
}

//DownloadImageSet: Resim veri setini internetten indirir ve belirtilen klasöre açar. URL'den zip dosyasını indirir, sıkıştırılmış dosyayı açar ve klasör adını döndürür.
string DownloadImageSet(string imagesDownloadFolder)
{
    string fileName = "flower_photos_small_set.zip";
    string url = $"https://aka.ms/mlnet-resources/datasets/flower_photos_small_set.zip";
    Web.Download(url, imagesDownloadFolder, fileName);
    Compress.UnZip(Path.Join(imagesDownloadFolder, fileName), imagesDownloadFolder);
    return Path.GetFileNameWithoutExtension(fileName);
}

//GetAbsolutePath: Göreceli bir yolu mutlak yola çevirir. Mevcut assembly'nin bulunduğu klasörü temel alarak göreceli yolu tam yola dönüştürür.
string GetAbsolutePath(string relativePath)
{
    FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
    string assemblyFolderPath = _dataRoot.Directory.FullName;
    string fullPath = Path.Combine(assemblyFolderPath, relativePath);
    return fullPath;
}
