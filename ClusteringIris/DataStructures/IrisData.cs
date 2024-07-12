namespace ClusteringIris.DataStructures;

//Bu sınıf, her bir iris çiçeğinin özelliklerini ve etiketini saklamak için kullanılır;
public class IrisData
{
    public float Label;//Bu alan, iris çiçeğinin türünü temsil eden bir etiket veya kategori değeridir. K-means algoritmasında etiketler kullanılmaz, çünkü algoritma gözetimsiz öğrenme algoritmasıdır, ancak değerlendirme veya veri setinin iç yapısını anlamak için etiketler mevcut olabilir.

    public float SepalLength;//Çiçeğin sepal (çanak yaprağı) uzunluğunu temsil eden bir değer.

    public float SepalWidth;//Çiçeğin sepal genişliğini temsil eden bir değer.

    public float PetalLength;//Çiçeğin petal (taç yaprağı) uzunluğunu temsil eden bir değer.

    public float PetalWidth;//Çiçeğin petal genişliğini temsil eden bir değer.
}
