using Microsoft.ML;
using Microsoft.ML.Transforms.TimeSeries;
using SpikeDetection.DataStructures;

//Yollar;
string BaseDatasetsRelativePath = @"C:\Users\firat\source\repos\MLDersleri\SpikeDetection\Data\";
string DatasetRelativePath = $"{BaseDatasetsRelativePath}/product-sales.csv";
string DatasetPath = GetAbsolutePath(DatasetRelativePath);
string BaseModelsRelativePath = @"C:\Users\firat\source\repos\MLDersleri\SpikeDetection\MLModels";
string ModelRelativePath = $"{BaseModelsRelativePath}/ProductSalesModel.zip";
string ModelPath = GetAbsolutePath(ModelRelativePath);

//MLContext nesnemiz;
MLContext mlContext = new MLContext();

//
const int size = 36;

//Veri Yükleme: LoadFromTextFile yöntemi, CSV dosyasından veri yükler. ProductSalesData sınıfı, CSV sütunlarını temsil eder;
IDataView dataView = mlContext.Data.LoadFromTextFile<ProductSalesData>(path: DatasetPath, hasHeader: true, separatorChar: ',');

//
DetectSpike(size, dataView);

//
DetectChangepoint(size, dataView);

//Çıktı;
Console.WriteLine("=============== End of process, hit any key to finish ===============");
Console.ReadLine();

//DetectSpike fonksiyonu, ML.NET kullanarak veri setindeki ani değişiklikleri tespit eder. IidSpikeEstimator kullanılarak bu değişiklikler belirlenir ve sonuçlar konsola yazdırılır.
void DetectSpike(int size, IDataView dataView)
{
    Console.WriteLine("===============Detect temporary changes in pattern===============");

    ////Spike tespiti için eğitici oluşturulması;
    IidSpikeEstimator estimator = mlContext.Transforms.DetectIidSpike
        (
        outputColumnName: nameof(ProductSalesPrediction.Prediction),
        inputColumnName: nameof(ProductSalesData.numSales),
        confidence: 95.0,
        pvalueHistoryLength: size / 4
        );

    //Modelin eğitilmesi;
    ITransformer tansformedModel = estimator.Fit(CreateEmptyDataView());

    //Verinin dönüştürülmesi;
    IDataView transformedData = tansformedModel.Transform(dataView);

    //Tahminlerin oluşturulması;
    IEnumerable<ProductSalesPrediction> predictions = mlContext.Data.CreateEnumerable<ProductSalesPrediction>(transformedData, reuseRowObject: false);

    //Sonuçların konsola yazdırılması
    Console.WriteLine("Alert\tScore\tP-Value");
    foreach (ProductSalesPrediction p in predictions)
    {
        //
        if (p.Prediction[0] == 1)
        {
            Console.BackgroundColor = ConsoleColor.DarkYellow;
            Console.ForegroundColor = ConsoleColor.Black;
        }
        Console.WriteLine("{0}\t{1:0.00}\t{2:0.00}", p.Prediction[0], p.Prediction[1], p.Prediction[2]);
        Console.ResetColor();
    }
    Console.WriteLine("");
    //Sonuçların konsola yazdırılması.
}

//DetectChangepoint fonksiyonu, ML.NET kullanarak veri setindeki sürekli değişiklikleri tespit eder. IidChangePointEstimator kullanılarak bu değişiklikler belirlenir ve sonuçlar konsola yazdırılır.
void DetectChangepoint(int size, IDataView dataView)
{
    Console.WriteLine("===============Detect Persistent changes in pattern===============");

    //Değişim noktası tespiti için eğitici oluşturulması
    IidChangePointEstimator estimator = mlContext.Transforms.DetectIidChangePoint
        (
        outputColumnName: nameof(ProductSalesPrediction.Prediction),
        inputColumnName: nameof(ProductSalesData.numSales),
        confidence: 95.0,
        changeHistoryLength: size / 4
        );

    //Modelin eğitilmesi;
    ITransformer tansformedModel = estimator.Fit(CreateEmptyDataView());

    //Verinin dönüştürülmesi;
    IDataView transformedData = tansformedModel.Transform(dataView);
    //Tahminlerin oluşturulması;
    IEnumerable<ProductSalesPrediction> predictions = mlContext.Data.CreateEnumerable<ProductSalesPrediction>(transformedData, reuseRowObject: false);

    //Sonuçların konsola yazdırılması
    Console.WriteLine($"{nameof(ProductSalesPrediction.Prediction)} column obtained post-transformation.");
    Console.WriteLine("Alert\tScore\tP-Value\tMartingale value");
    foreach (ProductSalesPrediction p in predictions)
    {
        if (p.Prediction[0] == 1)
        {
            Console.WriteLine("{0}\t{1:0.00}\t{2:0.00}\t{3:0.00}  <-- alert is on, predicted changepoint", p.Prediction[0], p.Prediction[1], p.Prediction[2], p.Prediction[3]);
        }
        else
        {
            Console.WriteLine("{0}\t{1:0.00}\t{2:0.00}\t{3:0.00}", p.Prediction[0], p.Prediction[1], p.Prediction[2], p.Prediction[3]);
        }
    }
    Console.WriteLine("");
    //Sonuçların konsola yazdırılması.
}

//GetAbsolutePath, programın çalıştığı dizin ve göreceli yol arasında bir mutlak yol oluşturur;
string GetAbsolutePath(string relativePath)
{
    FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
    string assemblyFolderPath = _dataRoot.Directory.FullName;
    string fullPath = Path.Combine(assemblyFolderPath, relativePath);
    return fullPath;
}

//CreateEmptyDataView, boş bir IDataView oluşturur ve ML.NET işlemleri için kullanılır;
IDataView CreateEmptyDataView()
{
    IEnumerable<ProductSalesData> enumerableData = new List<ProductSalesData>();
    IDataView dv = mlContext.Data.LoadFromEnumerable(enumerableData);
    return dv;
}