using Common;
using DatabaseIntegration;
using Microsoft.EntityFrameworkCore;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.LightGbm;
using Microsoft.ML.Transforms;
using System.Net;

string datasetUrl = "https://raw.githubusercontent.com/dotnet/machinelearning/244a8c2ac832657af282aa312d568211698790aa/test/data/adult.train";

IEnumerable<string> ReadRemoteDataset(string url)
{
    using (WebClient client = new WebClient())
    using (Stream stream = client.OpenRead(url))
    using (StreamReader reader = new StreamReader(stream))
    {
        string line;
        while ((line = reader.ReadLine()) != null)
        {
            yield return line;
        }
    }
}

IEnumerable<AdultCensus> QueryData()
{
    using (AdultCensusContext db = new AdultCensusContext())
    {
        foreach (var adult in db.AdultCensus.AsNoTracking().OrderBy(x => x.AdultCensusId))
        {
            yield return adult;
        }
    }
}

void CreateDatabase(string url)
{
    IEnumerable<string> dataset = ReadRemoteDataset(url);
    using (AdultCensusContext db = new AdultCensusContext())
    {
        db.Database.EnsureDeleted();
        db.Database.EnsureCreated();
        Console.WriteLine($"Database created, populating...");

        IEnumerable<AdultCensus> data = dataset.Skip(1).Select(l => l.Split(',')).Where(row => row.Length > 1).Select(row => new AdultCensus()
        {
            Age = int.Parse(row[0]),
            Workclass = row[1],
            Education = row[3],
            MaritalStatus = row[5],
            Occupation = row[6],
            Relationship = row[7],
            Race = row[8],
            Sex = row[9],
            CapitalGain = row[10],
            CapitalLoss = row[11],
            HoursPerWeek = int.Parse(row[12]),
            NativeCountry = row[13],
            Label = (int.Parse(row[14]) == 1) ? true : false
        });
        db.AdultCensus.AddRange(data);
        int count = db.SaveChanges();
        Console.WriteLine($"Total count of items saved to database: {count}");
    }
}

CreateDatabase(datasetUrl);

MLContext mlContext = new MLContext(seed: 1);

IDataView dataView = mlContext.Data.LoadFromEnumerable(QueryData());

DataOperationsCatalog.TrainTestData trainTestData = mlContext.Data.TrainTestSplit(dataView);

EstimatorChain<BinaryPredictionTransformer<CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator>>> pipeline = mlContext.Transforms.Categorical
    .OneHotEncoding(new[]
    {
        new InputOutputColumnPair("MsOHE", "MaritalStatus"),
        new InputOutputColumnPair("OccOHE", "Occupation"),
        new InputOutputColumnPair("RelOHE", "Relationship"),
        new InputOutputColumnPair("SOHE", "Sex"),
        new InputOutputColumnPair("NatOHE", "NativeCountry")
    }, OneHotEncodingEstimator.OutputKind.Binary)
    .Append(mlContext.Transforms.Concatenate("Features", "MsOHE", "OccOHE", "RelOHE", "SOHE", "NatOHE"))
    .Append(mlContext.BinaryClassification.Trainers.LightGbm());

Console.WriteLine("Training model...");

TransformerChain<BinaryPredictionTransformer<CalibratedModelParametersBase<LightGbmBinaryModelParameters, PlattCalibrator>>> model = pipeline.Fit(trainTestData.TrainSet);

Console.WriteLine("Predicting...");

IDataView predictions = model.Transform(trainTestData.TestSet);

CalibratedBinaryClassificationMetrics metrics = mlContext.BinaryClassification.Evaluate(predictions);
ConsoleHelper.PrintBinaryClassificationMetrics("Database Example", metrics);
ConsoleHelper.ConsolePressAnyKey();
