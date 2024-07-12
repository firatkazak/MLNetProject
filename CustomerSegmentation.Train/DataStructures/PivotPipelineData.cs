namespace CustomerSegmentation.Train.DataStructures;
public class PivotObservation
{
    public float[] Features;//Her bir müşteri için pivot verilerinden elde edilen özellikler.
    public string LastName;//Müşterinin soyadı.
}
//PivotObservation sınıfı, kümeleme modeli için gözlemleri temsil eder.