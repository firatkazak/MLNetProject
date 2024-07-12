using Common;
using CustomerSegmentation.Predict;
using Microsoft.ML;

//Dosya Yollarının Tanımlanması;
string assetsRelativePath = @"C:\Users\firat\source\repos\MLDersleri\CustomerSegmentation.Predict\Assets";
string assetsPath = GetAbsolutePath(assetsRelativePath);
string pivotCsv = Path.Combine(assetsPath, "Inputs", "pivot.csv");
string modelPath = Path.Combine(assetsPath, "Inputs", "retailClustering.zip");
string plotSvg = Path.Combine(assetsPath, "Outputs", "customerSegmentation.svg");
string plotCsv = Path.Combine(assetsPath, "Outputs", "customerSegmentation.csv");

try
{
    //ML.NET bağlamını oluşturur;
    MLContext mlContext = new MLContext();

    //Modeli kullanarak müşteri kümelerini oluşturmak için bir ClusteringModelScorer nesnesi oluşturur.
    ClusteringModelScorer clusteringModelScorer = new ClusteringModelScorer(mlContext, pivotCsv, plotSvg, plotCsv);

    //Modeli yükler.
    clusteringModelScorer.LoadModel(modelPath);

    //Müşteri kümelerini oluşturur.
    clusteringModelScorer.CreateCustomerClusters();
}
catch (Exception ex)
{
    //Herhangi bir hata olursa, hata mesajını konsola yazar.
    ConsoleHelper.ConsoleWriteException(ex.ToString());
}

//Kullanıcıdan bir tuşa basmasını bekler ve programın hemen kapanmasını engeller.
ConsoleHelper.ConsolePressAnyKey();

string GetAbsolutePath(string relativePath)
{
    FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);//Programın çalıştığı dosyanın yolunu alır.
    string assemblyFolderPath = _dataRoot.Directory.FullName;//Assembly'nin bulunduğu klasörün yolunu alır.
    string fullPath = Path.Combine(assemblyFolderPath, relativePath);//Göreceli yolu mutlak yola çevirir.
    return fullPath;//Mutlak yolu döner.
}
