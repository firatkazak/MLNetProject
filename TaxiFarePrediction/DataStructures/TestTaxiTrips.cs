namespace TaxiFarePrediction.DataStructures;
internal class SingleTaxiTripSample
{
    internal static readonly TaxiTrip Trip1 = new TaxiTrip
    {
        VendorId = "VTS",
        RateCode = "1",
        PassengerCount = 1,
        TripDistance = 10.33f,
        PaymentType = "CSH",
        FareAmount = 0//Tahmin edilecek değer, başlangıçta 0
    };
}
//Bu sınıf, tahmin işlemi için örnek bir taksi yolculuğu verisi sağlar. Modelinizi test ederken veya tahmin yaparken bu örnek kullanılabilir.

//Trip1: Örnek bir TaxiTrip nesnesi, modelimizi test etmek veya tahmin yapmak için kullanılır. Burada, FareAmount başlangıçta 0 olarak ayarlanmıştır çünkü model tarafından tahmin edilecektir.
