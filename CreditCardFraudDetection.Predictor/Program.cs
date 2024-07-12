using CCFraudDetection.Common;
using CreditCardFraudDetection.Predictor;

//Yollar;
string assetsPath = GetAbsolutePath(@"C:\Users\firat\source\repos\MLDersleri\CreditCardFraudDetection.Predictor\assets");
string trainOutput = GetAbsolutePath(@"C:\Users\firat\source\repos\MLDersleri\CreditCardFraudDetection.Trainer\assets\output");
string inputDatasetForPredictions = Path.Combine(assetsPath, "input", "testData.csv");
string modelFilePath = Path.Combine(assetsPath, "input", "randomizedPca.zip");

//Eğitim projesinden model ve veri setlerini kopyalar;
CopyModelAndDatasetFromTrainingProject(trainOutput, assetsPath);

//Predictor sınıfından yeni bir nesne oluşturur;
Predictor modelPredictor = new Predictor(modelFilePath, inputDatasetForPredictions);

//Modeli kullanarak 5 tahmin yapar;
modelPredictor.RunMultiplePredictions(numberOfPredictions: 5);

//Konsola bir mesaj yazdırır VE kullanıcıdan bir tuşa basmasını bekler.
Console.WriteLine("=============== Press any key ===============");
Console.ReadKey();

//Eğitim projesinden model ve veri setlerini kopyalamak için kullanılan metot.
void CopyModelAndDatasetFromTrainingProject(string trainOutput, string assetsPath)
{
    if (!File.Exists(Path.Combine(trainOutput, "testData.csv")) || !File.Exists(Path.Combine(trainOutput, "randomizedPca.zip")))
    {//Eğer testData.csv veya randomizedPca.zip dosyaları yoksa;
        Console.WriteLine("***** YOU NEED TO RUN THE TRAINING PROJECT FIRST *****");//Konsola eğitim projesini çalıştırmanız gerektiğini yazdır.
        Console.WriteLine("=============== Press any key ===============");//onsola bir mesaj yazdırır.
        Console.ReadKey();//Kullanıcıdan bir tuşa basmasını bekler.
        Environment.Exit(0);//Uygulamayı sonlandırır.
    }

    //assetsPath dizinini oluşturur.
    Directory.CreateDirectory(assetsPath);

    //Eğitim çıktılarındaki her dosya için,
    foreach (string file in Directory.GetFiles(trainOutput))
    {
        string fileDestination = Path.Combine(Path.Combine(assetsPath, "input"), Path.GetFileName(file));//Dosyanın hedef yolunu oluşturur.
        //Eğer dosya zaten varsa;
        if (File.Exists(fileDestination))
        {
            LocalConsoleHelper.DeleteAssets(fileDestination);//Dosyayı sil.
        }
        //Dosya testData.csv veya randomizedPca.zip ise
        if ((Path.GetFileName(file) == "testData.csv") || (Path.GetFileName(file) == "randomizedPca.zip"))
            File.Copy(file, Path.Combine(Path.Combine(assetsPath, "input"), Path.GetFileName(file)));//Dosyayı kopyala.
    }
}

//Path metodu;
string GetAbsolutePath(string relativePath)
{
    FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
    string assemblyFolderPath = _dataRoot.Directory.FullName;
    string fullPath = Path.Combine(assemblyFolderPath, relativePath);
    return fullPath;
}
