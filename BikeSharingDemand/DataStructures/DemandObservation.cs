using Microsoft.ML.Data;

namespace BikeSharingDemand.DataStructures;
public class DemandObservation
{
    [LoadColumn(2)]//LoadColumn niteliği: Bu nitelik, veri dosyasındaki sütun indeksini belirtir. Örneğin, Season özelliği 2 numaralı sütunu (LoadColumn(2)) temsil eder.
    public float Season { get; set; }
    [LoadColumn(3)]
    public float Year { get; set; }
    [LoadColumn(4)]
    public float Month { get; set; }
    [LoadColumn(5)]
    public float Hour { get; set; }
    [LoadColumn(6)]
    public float Holiday { get; set; }
    [LoadColumn(7)]
    public float Weekday { get; set; }
    [LoadColumn(8)]
    public float WorkingDay { get; set; }
    [LoadColumn(9)]
    public float Weather { get; set; }
    [LoadColumn(10)]
    public float Temperature { get; set; }
    [LoadColumn(11)]
    public float NormalizedTemperature { get; set; }
    [LoadColumn(12)]
    public float Humidity { get; set; }
    [LoadColumn(13)]
    public float Windspeed { get; set; }
    [LoadColumn(16)]
    [ColumnName("Label")]
    public float Count { get; set; }//Count özelliği: Bu özellik, talep sayısını temsil eder ve Label adı altında ML.NET'e girdi etiketini bildirir.
}

public static class DemandObservationSample
{
    //SingleDemandSampleData: Bu özellik, bir örneğin sabit veriye sahip bir örneğini temsil eder.
    //Örneğin, burada belirtilen değerler DemandObservation sınıfının özelliklerine karşılık gelir.
    public static DemandObservation SingleDemandSampleData => new DemandObservation()
    {
        Season = 3,
        Year = 1,
        Month = 8,
        Hour = 10,
        Holiday = 0,
        Weekday = 4,
        WorkingDay = 1,
        Weather = 1,
        Temperature = 0.8f,
        NormalizedTemperature = 0.7576f,
        Humidity = 0.55f,
        Windspeed = 0.2239f
    };
}
