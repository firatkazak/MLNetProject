using CCFraudDetection.Common;
using CCFraudDetection.Predictor;

string assetsPath = GetAbsolutePath(@"C:\Users\firat\source\repos\MLDersleri\CCFraudDetection.Predictor\Assets");
string trainOutput = GetAbsolutePath(@"C:\Users\firat\source\repos\MLDersleri\CCFraudDetection.Predictor\Assets\Output");
//Bu 2 değişken model dosyalarının ve veri setlerinin bulunduğu dosya yolunu içerir. GetAbsolutePath metodunu kullanarak mutlak dosya yolunu elde eder.
CopyModelAndDatasetFromTrainingProject(trainOutput, assetsPath);
//Eğitim projesinden model dosyalarını ve veri setlerini bu projeye kopyalar.
string inputDatasetForPredictions = Path.Combine(assetsPath, "Input", "testData.csv");
//Tahminlerin yapılacağı veri setinin yolunu belirtir.
string modelFilePath = Path.Combine(assetsPath, "Input", "fastTree.zip");
//Kullanılacak model dosyasının yolunu belirtir.
Predictor modelPredictor = new Predictor(modelFilePath, inputDatasetForPredictions);
//Belirtilen model dosyasını ve veri setini kullanarak tahminler yapmak için Predictor sınıfından bir örnek aldık.
modelPredictor.RunMultiplePredictions(numberOfPredictions: 5);
//Belirtilen sayıda (burada 5) tahmin yapılır ve sonuçlar ekrana yazdırılır.
Console.WriteLine("=============== Press any key ===============");
Console.ReadKey();
//Kullanıcıya bir tuşa basması gerektiği mesajı verilir ve herhangi bir tuşa basılana kadar programın beklemesini sağlar.

//CopyModelAndDatasetFromTrainingProject(): Eğitim projesinden model dosyalarını ve veri setlerini bu projeye kopyalamak için kullanılır.
void CopyModelAndDatasetFromTrainingProject(string trainOutput, string assetsPath)
{
    if (!File.Exists(Path.Combine(trainOutput, "testData.csv")) || !File.Exists(Path.Combine(trainOutput, "fastTree.zip")))
    {//Eğer gerekli dosyalar (testData.csv ve fastTree.zip) trainOutput dizininde bulunmuyorsa, kullanıcıya uygun mesaj verilir ve program sonlandırılır.
        Console.WriteLine("***** YOU NEED TO RUN THE TRAINING PROJECT IN THE FIRST PLACE *****");
        Console.WriteLine("=============== Press any key ===============");
        Console.ReadKey();
        Environment.Exit(0);
    }

    //Eğer gerekli dosyalar (testData.csv ve fastTree.zip) trainOutput dizininde bulunuyorsa dosyalar hedef dizine (assetsPath) kopyalanır.
    Directory.CreateDirectory(assetsPath);
    foreach (string file in Directory.GetFiles(trainOutput))
    {
        string fileDestination = Path.Combine(Path.Combine(assetsPath, "Input"), Path.GetFileName(file));
        if (File.Exists(fileDestination))
        {
            LocalConsoleHelper.DeleteAssets(fileDestination);
        }
        File.Copy(file, Path.Combine(Path.Combine(assetsPath, "Input"), Path.GetFileName(file)));
    }
}

//Verilen göreceli yolun mutlak yolunu döndürür.
string GetAbsolutePath(string relativePath)//relativePath parametresi, göreceli yolun bir dize olarak temsilidir.
{
    FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
    string assemblyFolderPath = _dataRoot.Directory.FullName;
    string fullPath = Path.Combine(assemblyFolderPath, relativePath);
    return fullPath;
}//Mevcut programın çalıştığı klasör (Assembly.Location) kullanılarak mutlak yol oluşturulur ve döndürülür.
