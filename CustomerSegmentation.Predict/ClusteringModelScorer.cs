using Common;
using CustomerSegmentation.Train.DataStructures;
using Microsoft.ML;
using Microsoft.ML.Data;
using OxyPlot;
using OxyPlot.Series;
using System.Diagnostics;

namespace CustomerSegmentation.Predict;

//Bu sınıf, eğitimli modeli kullanarak müşteri kümelerini oluşturur ve sonuçları CSV dosyasına ve grafik dosyasına kaydeder;
public class ClusteringModelScorer
{
    private readonly string _pivotDataLocation;//Dosya yollarını tutan değişkenler.
    private readonly string _plotLocation;//Dosya yollarını tutan değişkenler.
    private readonly string _csvlocation;//Dosya yollarını tutan değişkenler.
    private readonly MLContext _mlContext;//ML.NET bağlamını tutan değişken.
    private ITransformer _trainedModel;//Eğitimli modeli tutan değişken.

    public ClusteringModelScorer(MLContext mlContext, string pivotDataLocation, string plotLocation, string csvlocation)
    {
        _pivotDataLocation = pivotDataLocation;
        _plotLocation = plotLocation;
        _csvlocation = csvlocation;
        _mlContext = mlContext;
    }

    //Belirtilen dosya yolundan modeli yükler;
    public ITransformer LoadModel(string modelPath)
    {
        //modelPath: Model dosyasının yolu. modelInputSchema: Modelin girdi şemasını döndürür.
        _trainedModel = _mlContext.Model.Load(modelPath, out DataViewSchema modelInputSchema);
        return _trainedModel;//Yüklenen modeli döndürür.
    }

    //Müşteri kümelerini oluşturur;
    public void CreateCustomerClusters()
    {
        //Veriyi yükler ve sütunları tanımlar (Features ve LastName).
        IDataView data = _mlContext.Data.LoadFromTextFile(path: _pivotDataLocation, columns: new[]
        {
            new TextLoader.Column("Features", DataKind.Single, new[] {new TextLoader.Range(0, 31) }),
            new TextLoader.Column(nameof(PivotData.LastName), DataKind.String, 32)
        }, hasHeader: true, separatorChar: ',');

        //Veriyi modelle dönüştürür;
        IDataView tranfomedDataView = _trainedModel.Transform(data);

        //Tahminleri oluşturur ve bir diziye dönüştürür.
        ClusteringPrediction[] predictions = _mlContext.Data.CreateEnumerable<ClusteringPrediction>(tranfomedDataView, false).ToArray();

        //Tahminleri CSV dosyasına kaydeder.
        SaveCustomerSegmentationCSV(predictions, _csvlocation);

        //Tahminleri grafik dosyasına kaydeder.
        SaveCustomerSegmentationPlotChart(predictions, _plotLocation);

        //Grafiği varsayılan pencere ile açar.
        OpenChartInDefaultWindow(_plotLocation);
    }

    //Tahminleri CSV dosyasına kaydeder;
    private static void SaveCustomerSegmentationCSV(IEnumerable<ClusteringPrediction> predictions, string csvlocation)
    {
        ConsoleHelper.ConsoleWriteHeader("CSV Customer Segmentation");//Başlığı yazar.

        using (StreamWriter w = new StreamWriter(csvlocation))//CSV dosyasını açar.
        {
            w.WriteLine($"LastName,SelectedClusterId");//Başlık satırını yazar.
            w.Flush();//Yazma işlemini tamamlar.

            ////Her bir tahmini yazar;
            predictions.ToList().ForEach(prediction =>
            {
                w.WriteLine($"{prediction.LastName},{prediction.SelectedClusterId}");
                w.Flush();//Yazma işlemini tamamlar.
            });
        }

        Console.WriteLine($"CSV location: {csvlocation}");
    }

    //Tahminleri grafik dosyasına kaydeder;
    private static void SaveCustomerSegmentationPlotChart(IEnumerable<ClusteringPrediction> predictions, string plotLocation)
    {
        ConsoleHelper.ConsoleWriteHeader("Plot Customer Segmentation");//Başlığı yazar.

        //Grafik modelini oluşturur;
        PlotModel plot = new PlotModel { Title = "Customer Segmentation", IsLegendVisible = true };

        //Kümeleri belirler;
        IOrderedEnumerable<uint> clusters = predictions.Select(p => p.SelectedClusterId).Distinct().OrderBy(x => x);

        //Her bir küme için;
        foreach (uint cluster in clusters)
        {
            //Dağılım serisini oluşturur;
            ScatterSeries scatter = new ScatterSeries { MarkerType = MarkerType.Circle, MarkerStrokeThickness = 2, Title = $"Cluster: {cluster}", RenderInLegend = true };

            //Bu satır, LINQ ifadeleri kullanarak predictions dizisinden belirli bir küme (cluster) için ScatterPoint dizisi oluşturur.
            ScatterPoint[] series = predictions.Where(p => p.SelectedClusterId == cluster).Select(p => new ScatterPoint(p.Location[0], p.Location[1])).ToArray();

            //Noktaları seriye ekler.
            scatter.Points.AddRange(series);

            //Seriyi grafiğe ekler.
            plot.Series.Add(scatter);
        }

        //Grafik renklerini ayarlar;
        plot.DefaultColors = OxyPalettes.HueDistinct(plot.Series.Count).Colors;

        //SVG dışa aktarıcıyı oluşturur;
        SvgExporter exporter = new SvgExporter { Width = 600, Height = 400 };

        //Dosya akışını açar;
        using (FileStream fs = new FileStream(plotLocation, FileMode.Create))
        {
            exporter.Export(plot, fs);//Grafiği dışa aktarır.
        }

        Console.WriteLine($"Plot location: {plotLocation}");//Grafik dosyasının yolunu yazar.
    }

    private static void OpenChartInDefaultWindow(string plotLocation)
    {
        Console.WriteLine("Showing chart...");//Mesajı yazar.
        Process p = new Process();//Yeni bir süreç oluşturur.
        p.StartInfo = new ProcessStartInfo(plotLocation)//Süreç başlatma bilgisini ayarlar.
        {
            UseShellExecute = true//ShellExecute özelliğini etkinleştirir.
        };
        p.Start();//Süreci başlatır.
    }
}
