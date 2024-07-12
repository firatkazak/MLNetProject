namespace ImageClassification.Train.Model;
public static class ModelHelpers
{
    //FileInfo _dataRoot = new FileInfo(typeof(ModelHelpers).Assembly.Location);: Sınıfın bulunduğu dizini temsil eden bir FileInfo nesnesi oluşturur.
    static FileInfo _dataRoot = new FileInfo(typeof(ModelHelpers).Assembly.Location);

    //Verilen dosya yollarını _dataRoot dizini ile birleştirerek tam dosya yolunu döndürür.
    public static string GetAssetsPath(params string[] paths)
    {
        if (paths == null || paths.Length == 0)
            return null;

        return Path.Combine(paths.Prepend(_dataRoot.Directory.FullName).ToArray());
    }

    //GetAssetsPath metodunu kullanarak tam dosya yolunu alır ve eğer dosya varsa siler. Dosya yolunu döndürür.
    public static string DeleteAssets(params string[] paths)
    {
        string location = GetAssetsPath(paths);

        if (!string.IsNullOrWhiteSpace(location) && File.Exists(location))
            File.Delete(location);
        return location;
    }

    //En yüksek olasılığa sahip etiketi ve bu olasılığı döndürür. probs dizisindeki en yüksek değeri bulur ve bu değerin dizinini kullanarak labels dizisinden etiketi alır.
    public static (string, float) GetLabel(string[] labels, float[] probs)
    {
        float max = probs.Max();
        int index = probs.AsSpan().IndexOf(max);
        return (labels[index], max);
    }

    //T türündeki float ve int türündeki özellik adlarını dizi olarak döndürür.
    public static string[] ColumnsNumerical<T>() where T : class
    {
        return Columns<T, float, int>().ToArray();
    }

    //T türündeki string türündeki özellik adlarını dizi olarak döndürür.
    public static string[] ColumnsString<T>() where T : class
    {
        return Columns<T, string>().ToArray();
    }

    // T türündeki DateTime türündeki özellik adlarını dizi olarak döndürür;
    public static string[] ColumnsDateTime<T>() where T : class
    {
        return Columns<T, DateTime>().ToArray();
    }

    //Columns<T>(): T sınıfının (class) tüm özellik (property) isimlerini döndürür.
    public static IEnumerable<string> Columns<T>() where T : class
    {
        return typeof(T).GetProperties().Select(p => p.Name);
    }

    //Columns<T, U>(): Bu metot, T sınıfının yalnızca U türündeki özellik isimlerini döndürür.
    public static IEnumerable<string> Columns<T, U>() where T : class
    {
        Type typeofU = typeof(U);
        return typeof(T).GetProperties().Where(c => c.PropertyType == typeofU).Select(p => p.Name);
    }

    //Columns<T, U, V>(): Bu metot, T sınıfının U veya V türündeki özellik isimlerini döndürür.
    public static IEnumerable<string> Columns<T, U, V>() where T : class
    {
        Type[] typeofUV = new[] { typeof(U), typeof(V) };
        return typeof(T).GetProperties().Where(c => typeofUV.Contains(c.PropertyType)).Select(p => p.Name);
    }

    //Columns<T, U, V, W>(): Bu metot, T sınıfının U, V veya W türündeki özellik isimlerini döndürür.
    public static IEnumerable<string> Columns<T, U, V, W>() where T : class
    {
        Type[] typeofUVW = new[] { typeof(U), typeof(V), typeof(W) };
        return typeof(T).GetProperties().Where(c => typeofUVW.Contains(c.PropertyType)).Select(p => p.Name);
    }
}
