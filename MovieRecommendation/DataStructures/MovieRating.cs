using Microsoft.ML.Data;

namespace MovieRecommendation.DataStructures;
public class MovieRating//Film derecelendirmelerinin temsil edildiği bir sınıftır. recommendation-ratings-train.csv ve recommendation-ratings-test.csv dosyalarından veri yükler.
{
    [LoadColumn(0)]//Veri yükleme işlemleri için kullanılır.
    public float userId;

    [LoadColumn(1)]
    public float movieId;

    [LoadColumn(2)]
    public float Label;
}
//userId, movieId ve Label özellikleri ile her bir derecelendirmeyi tanımlar.
