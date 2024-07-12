using BikeSharingDemand.DataStructures;
using Common;
using Microsoft.ML;
using System.Globalization;

namespace BikeSharingDemand;
public static class ModelScoringTester
{
    //VisualizeSomePredictions Metodu: Bu metod, modelin bazı tahminlerini görselleştirmek için kullanılır. Verilen testDataLocation yolundaki CSV dosyasından veri okur, her bir veri örneği için tahmin yapar ve sonucu konsola basar.
    public static void VisualizeSomePredictions(MLContext mlContext, string modelName, string testDataLocation, PredictionEngine<DemandObservation, DemandPrediction> predEngine, int numberOfPredictions)
    {
        List<DemandObservation> testData = ReadSampleDataFromCsvFile(testDataLocation, numberOfPredictions);
        for (int i = 0; i < numberOfPredictions; i++)
        {
            DemandPrediction resultprediction = predEngine.Predict(testData[i]);
            ConsoleHelper.PrintRegressionPredictionVersusObserved(resultprediction.PredictedCount.ToString(),
                testData[i].Count.ToString());
        }

    }

    //ReadSampleDataFromCsvFile Metodu: Bu metod, CSV dosyasından örnek verileri okur. Dosyayı satır satır okur, belirli sütunlardaki verileri DemandObservation nesnelerine dönüştürür.
    public static List<DemandObservation> ReadSampleDataFromCsvFile(string dataLocation, int numberOfRecordsToRead)
    {
        return File.ReadLines(dataLocation)
            .Skip(1)
            .Where(x => !string.IsNullOrWhiteSpace(x))
            .Select(x => x.Split(','))
            .Select(x => new DemandObservation()
            {
                Season = float.Parse(x[2], CultureInfo.InvariantCulture),
                Year = float.Parse(x[3], CultureInfo.InvariantCulture),
                Month = float.Parse(x[4], CultureInfo.InvariantCulture),
                Hour = float.Parse(x[5], CultureInfo.InvariantCulture),
                Holiday = float.Parse(x[6], CultureInfo.InvariantCulture),
                Weekday = float.Parse(x[7], CultureInfo.InvariantCulture),
                WorkingDay = float.Parse(x[8], CultureInfo.InvariantCulture),
                Weather = float.Parse(x[9], CultureInfo.InvariantCulture),
                Temperature = float.Parse(x[10], CultureInfo.InvariantCulture),
                NormalizedTemperature = float.Parse(x[11], CultureInfo.InvariantCulture),
                Humidity = float.Parse(x[12], CultureInfo.InvariantCulture),
                Windspeed = float.Parse(x[13], CultureInfo.InvariantCulture),
                Count = float.Parse(x[16], CultureInfo.InvariantCulture)
            })
            .Take(numberOfRecordsToRead)
            .ToList();
    }
}
