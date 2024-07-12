namespace CreditCardFraudDetection.Common;

public static class LocalConsoleHelper
{
    //Bu metot, bir veya daha fazla dosya yolunu alır ve bunları birleşik bir yol olarak döner;
    public static string GetAssetsPath(params string[] paths)
    {
        FileInfo _dataRoot = new FileInfo(typeof(LocalConsoleHelper).Assembly.Location);//çalışmakta olan derlemenin bulunduğu dizinin yolunu alır.
        if (paths == null || paths.Length == 0)//Eğer paths null veya boş ise;
        {
            return null;//null döner.
        }
        return Path.Combine(paths.Prepend(_dataRoot.Directory.FullName).ToArray());//_dataRoot yolunu paths'in başına ekler ve tüm yolları birleştirir.
    }

    //Bu metot, belirtilen dosya yolunu siler;
    public static string DeleteAssets(params string[] paths)
    {
        string location = GetAssetsPath(paths);//GetAssetsPath metodunu çağırarak birleşik yolu alır.
        if (!string.IsNullOrWhiteSpace(location) && File.Exists(location))//Eğer location boş değilse ve dosya mevcutsa;
        {
            File.Delete(location);//Dosyayı siler.
        }
        return location;//Silinen dosyanın yolunu döner.
    }
}
