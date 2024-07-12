using ImageClassification.TensorFlow.ModelScorer;

//Yollar;
string assetsRelativePath = @"C:\Users\firat\source\repos\MLDersleri\ImageClassification.TensorFlow\Assets";
string assetsPath = GetAbsolutePath(assetsRelativePath);
string tagsTsv = Path.Combine(assetsPath, "Inputs", "Images", "tags.tsv");
string imagesFolder = Path.Combine(assetsPath, "Inputs", "Images");
string inceptionPb = Path.Combine(assetsPath, "Inputs", "Inception", "tensorflow_inception_graph.pb");
string labelsTxt = Path.Combine(assetsPath, "Inputs", "Inception", "imagenet_comp_graph_label_strings.txt");

try
{
    //TFModelScorer sınıfından bir örnek (modelScorer) oluşturulur ve TensorFlow modelinin yolunu, görüntü klasörünü ve etiket dosyasının yolunu parametre olarak alır.
    TFModelScorer modelScorer = new TFModelScorer(tagsTsv, imagesFolder, inceptionPb, labelsTxt);

    //modelScorer.Score() metoduyla modelin çalıştırılması ve görüntülerin sınıflandırılması işlemi başlatılır.
    modelScorer.Score();

}
catch (Exception ex)
{
    //Eğer bir hata oluşursa, ConsoleHelpers.ConsoleWriteException metoduyla konsola istisna (exception) ayrıntıları yazdırılır.
    ConsoleHelpers.ConsoleWriteException(ex.ToString());
}

//ConsoleHelper metodu;
ConsoleHelpers.ConsolePressAnyKey();

//Path metodu;
string GetAbsolutePath(string relativePath)
{
    FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
    string assemblyFolderPath = _dataRoot.Directory.FullName;
    string fullPath = Path.Combine(assemblyFolderPath, relativePath);
    return fullPath;
}
