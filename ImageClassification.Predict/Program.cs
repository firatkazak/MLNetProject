using ImageClassification.Predict.Model;
using ImageClassification.Train.Model;

//Yollar;
string assetsRelativePath = @"C:\Users\firat\source\repos\MLDersleri\ImageClassification.Predict\assets";
string assetsPath = GetAbsolutePath(assetsRelativePath);
string imagesFolder = Path.Combine(assetsPath, "inputs", "images-for-predictions");
string imageClassifierZip = Path.Combine(assetsPath, "inputs", "MLNETModel", "imageClassifier.zip");

try
{
    //ModelScorer sınıfının bir örneği oluşturulur.
    ModelScorer modelScorer = new ModelScorer(imagesFolder, imageClassifierZip);
    //ClassifyImages metodu çağrılarak resimlerin tahminleri yapılır.
    modelScorer.ClassifyImages();
}
catch (Exception ex)
{
    //Eğer bir hata meydana gelirse, bu hata yakalanır ve konsola yazdırılır.
    ConsoleHelpers.ConsoleWriteException(ex.ToString());
}

//Kullanıcıdan herhangi bir tuşa basmasını isteyen bir mesaj görüntülenir, böylece konsol penceresi hemen kapanmaz.
ConsoleHelpers.ConsolePressAnyKey();

//GetAbsolutePath: Verilen göreceli yolu(relative path) tam bir yol(absolute path) haline dönüştürür.
//Program sınıfının bulunduğu klasörün tam yolunu alır ve buna göreceli yolu ekler.
string GetAbsolutePath(string relativePath)
{
    FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
    string assemblyFolderPath = _dataRoot.Directory.FullName;
    string fullPath = Path.Combine(assemblyFolderPath, relativePath);
    return fullPath;
}
