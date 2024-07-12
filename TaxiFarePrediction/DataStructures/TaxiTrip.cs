using Microsoft.ML.Data;

namespace TaxiFarePrediction.DataStructures;
public class TaxiTrip
{
    [LoadColumn(0)]
    public string VendorId;//Taksi şirketinin kimliği

    [LoadColumn(1)]
    public string RateCode;//Ücret kodu

    [LoadColumn(2)]
    public float PassengerCount;//Yolcu sayısı

    [LoadColumn(3)]
    public float TripTime;//Yolculuk süresi

    [LoadColumn(4)]
    public float TripDistance;//Yolculuk mesafesi

    [LoadColumn(5)]
    public string PaymentType;//Ödeme türü

    [LoadColumn(6)]
    public float FareAmount;//Gerçek taksi ücreti (hedef değişken)
}
//Bu sınıf, taksi yolculuğuna ait verilerin yapısını tanımlar. Verilerimizin her bir sütunu bu sınıfta bir property olarak tanımlanmış.