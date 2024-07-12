using Microsoft.ML.Data;

namespace TaxiFarePrediction.DataStructures;
public class TaxiTripFarePrediction
{
    [ColumnName("Score")]//ColumnName("Score") attribute'u, ML.NET modelinin tahmin edilen değeri bu property'ye atamasını sağlar.

    public float FareAmount;//FareAmount: Modelin tahmin ettiği taksi ücreti.
}
//Bu sınıf, modelimizin tahmin ettiği sonuçları tutmak için kullanılır.