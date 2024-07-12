using System;

namespace ImageClassification.TensorFlow.ModelScorer;

//ModelHelpers: Modelleme ve veri işleme süreçlerinde yardımcı olmak için tasarlanmış bir sınıf;
public static class ModelHelpers
{
    //Program sınıfının mevcut yürütülebilir dosyasının yolunu temsil eder;
    static FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);

    //GetAssetsPath: Belirtilen yol dizinlerini _dataRoot'un dizin adıyla birleştirerek döndürür.
    public static string GetAssetsPath(params string[] paths)
    {
        if (paths == null || paths.Length == 0)
            return null;

        return Path.Combine(paths.Prepend(_dataRoot.Directory.FullName).ToArray());
    }

    //DeleteAssets: GetAssetsPath metodunu kullanarak belirtilen dosya yolunu alır. Yol varsa ve dosya mevcutsa siler, aksi halde null döner.
    public static string DeleteAssets(params string[] paths)
    {
        string location = GetAssetsPath(paths);

        if (!string.IsNullOrWhiteSpace(location) && File.Exists(location))
            File.Delete(location);
        return location;
    }

    //GetBestLabel: Verilen olasılık dizisi içinden en yüksek olasılığı bulur. Bu en yüksek olasılığa karşılık gelen etiketi ve olasılığı bir tuple olarak döndürür.
    public static (string, float) GetBestLabel(string[] labels, float[] probs)
    {
        float max = probs.Max();
        int index = probs.AsSpan().IndexOf(max);
        return (labels[index], max);
    }

    //ReadLabels: Belirtilen etiket dosyasından etiketleri okur ve döndürür.
    public static string[] ReadLabels(string labelsLocation)
    {
        return File.ReadAllLines(labelsLocation);
    }

    //Columns<T>(): Generic tür T için sınıfın özellik isimlerini döndürür.
    public static IEnumerable<string> Columns<T>() where T : class
    {
        return typeof(T).GetProperties().Select(p => p.Name);
    }

    //Columns<T, U>(): Tür T'de, U türündeki özellik isimlerini döndürür.
    public static IEnumerable<string> Columns<T, U>() where T : class
    {
        Type typeofU = typeof(U);
        return typeof(T).GetProperties().Where(c => c.PropertyType == typeofU).Select(p => p.Name);
    }

    //Columns<T, U, V>(): T türünde, U veya V türlerindeki özellik isimlerini döndürür.
    public static IEnumerable<string> Columns<T, U, V>() where T : class
    {
        Type[] typeofUV = new[] { typeof(U), typeof(V) };
        return typeof(T).GetProperties().Where(c => typeofUV.Contains(c.PropertyType)).Select(p => p.Name);
    }

    //Columns<T, U, V, W>(): T türünde, U, V veya W türlerindeki özellik isimlerini döndürür.
    public static IEnumerable<string> Columns<T, U, V, W>() where T : class
    {
        Type[] typeofUVW = new[] { typeof(U), typeof(V), typeof(W) };
        return typeof(T).GetProperties().Where(c => typeofUVW.Contains(c.PropertyType)).Select(p => p.Name);
    }

    //ColumnsNumerical<T>(): Belirtilen tür için sayısal özelliklerinin isimlerini döndürür.
    public static string[] ColumnsNumerical<T>() where T : class
    {
        return Columns<T, float, int>().ToArray();
    }

    //ColumnsString<T>(): Belirtilen tür için dize özelliklerinin isimlerini döndürür.
    public static string[] ColumnsString<T>() where T : class
    {
        return Columns<T, string>().ToArray();
    }

    //ColumnsDateTime<T>(): Belirtilen tür için tarih özelliklerinin isimlerini döndürür.
    public static string[] ColumnsDateTime<T>() where T : class
    {
        return Columns<T, DateTime>().ToArray();
    }
}
