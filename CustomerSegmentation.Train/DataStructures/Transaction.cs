namespace CustomerSegmentation.Train.DataStructures;

//Transaction sınıfı, işlemleri temsil eder;
public class Transaction
{
    public string LastName { get; set; }//Müşterinin soyadı.
    public string OfferId { get; set; }//İşlem yapılan teklifin ID'si.

    public static IEnumerable<Transaction> ReadFromCsv(string file)//CSV dosyasından işlem verilerini okur.
    {
        return File.ReadAllLines(file).Skip(1).Select(x => x.Split(',')).Select(x => new Transaction()
        {
            LastName = x[0],
            OfferId = x[1],
        });
    }
}
