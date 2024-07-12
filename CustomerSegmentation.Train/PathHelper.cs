namespace CustomerSegmentation.Train;

//PathHelper sınıfı, dosya yolları ile ilgili yardımcı metotlar sağlar;
public static class PathHelper
{
    static FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);

    //Belirtilen dosya yollarını birleştirerek tam bir dosya yolu oluşturur;
    public static string GetAssetsPath(params string[] paths)
    {
        if (paths == null || paths.Length == 0)
            return null;

        return Path.Combine(paths.Prepend(_dataRoot.Directory.FullName).ToArray());
    }

    //Belirtilen dosya yollarındaki dosyayı siler;
    public static string DeleteAssets(params string[] paths)
    {
        string location = GetAssetsPath(paths);

        if (!string.IsNullOrWhiteSpace(location) && File.Exists(location))
            File.Delete(location);

        return location;
    }

}