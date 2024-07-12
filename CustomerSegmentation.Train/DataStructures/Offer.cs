using System.Globalization;

namespace CustomerSegmentation.Train.DataStructures;
//Offer sınıfı, teklifler (offers) ile ilgili verileri temsil eder:
//OfferId, Campaign, Varietal, Minimum, Discount, Origin, LastPeak: Tekliflerin çeşitli özelliklerini tutar.
public class Offer
{
    public string OfferId { get; set; }
    public string Campaign { get; set; }
    public string Varietal { get; set; }
    public float Minimum { get; set; }
    public float Discount { get; set; }
    public string Origin { get; set; }
    public string LastPeak { get; set; }

    //ReadFromCsv: CSV dosyasından teklif verilerini okur;
    public static IEnumerable<Offer> ReadFromCsv(string file)
    {
        return File.ReadAllLines(file).Skip(1).Select(x => x.Split(',')).Select(x => new Offer()
        {
            OfferId = x[0],
            Campaign = x[1],
            Varietal = x[2],
            Minimum = float.Parse(x[3], CultureInfo.InvariantCulture),
            Discount = float.Parse(x[4], CultureInfo.InvariantCulture),
            Origin = x[5],
            LastPeak = x[6]
        });
    }
}
