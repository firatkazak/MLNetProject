using System.Diagnostics;
using System.Globalization;
using Common;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using PLplot;
using TaxiFarePrediction.DataStructures;

//Alınan Path'ler;
string AppPath = Path.GetDirectoryName(Environment.GetCommandLineArgs()[0]);
string BaseDatasetsRelativePath = @"C:\Users\firat\source\repos\MLDersleri\TaxiFarePrediction\Data";
string TrainDataRelativePath = $"{BaseDatasetsRelativePath}/taxi-fare-train.csv";
string TestDataRelativePath = $"{BaseDatasetsRelativePath}/taxi-fare-test.csv";
string TrainDataPath = GetAbsolutePath(TrainDataRelativePath);
string TestDataPath = GetAbsolutePath(TestDataRelativePath);
string BaseModelsRelativePath = @"C:\Users\firat\source\repos\MLDersleri\TaxiFarePrediction\Model";
string ModelRelativePath = $"{BaseModelsRelativePath}/TaxiFareModel.zip";
string ModelPath = GetAbsolutePath(ModelRelativePath);
//Path'ler bitiş.

//ML.NET operasyonlarının gerçekleştirildiği bağlam.
MLContext mlContext = new MLContext(seed: 0);

//Modelin eğitilmesi, değerlendirilmesi ve kaydedilmesi işlemlerini gerçekleştirir.
BuildTrainEvaluateAndSaveModel(mlContext);

//Tek bir örnek üzerinde tahmin yapar.
TestSinglePrediction(mlContext);

//Tahmin sonuçlarını grafik olarak gösterir.
PlotRegressionChart(mlContext, TestDataPath, 100, args);

Console.WriteLine("Press any key to exit..");
Console.ReadLine();

ITransformer BuildTrainEvaluateAndSaveModel(MLContext mlContext)
{
    //Veri yükleme
    IDataView baseTrainingDataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(TrainDataPath, hasHeader: true, separatorChar: ',');
    IDataView testDataView = mlContext.Data.LoadFromTextFile<TaxiTrip>(TestDataPath, hasHeader: true, separatorChar: ',');

    int cnt = baseTrainingDataView.GetColumn<float>(nameof(TaxiTrip.FareAmount)).Count();

    //Veri filtreleme
    IDataView trainingDataView = mlContext.Data.FilterRowsByColumn(baseTrainingDataView, nameof(TaxiTrip.FareAmount), lowerBound: 1, upperBound: 150);
    int cnt2 = trainingDataView.GetColumn<float>(nameof(TaxiTrip.FareAmount)).Count();

    //Veri işleme pipeline'ı oluşturma
    EstimatorChain<ColumnConcatenatingTransformer> dataProcessPipeline = mlContext.Transforms
        .CopyColumns(outputColumnName: "Label", inputColumnName: nameof(TaxiTrip.FareAmount))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "VendorIdEncoded", inputColumnName: nameof(TaxiTrip.VendorId)))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "RateCodeEncoded", inputColumnName: nameof(TaxiTrip.RateCode)))
        .Append(mlContext.Transforms.Categorical.OneHotEncoding(outputColumnName: "PaymentTypeEncoded", inputColumnName: nameof(TaxiTrip.PaymentType)))
        .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(TaxiTrip.PassengerCount)))
        .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(TaxiTrip.TripTime)))
        .Append(mlContext.Transforms.NormalizeMeanVariance(outputColumnName: nameof(TaxiTrip.TripDistance)))
        .Append(mlContext.Transforms.Concatenate("Features", "VendorIdEncoded", "RateCodeEncoded", "PaymentTypeEncoded", nameof(TaxiTrip.PassengerCount), nameof(TaxiTrip.TripTime), nameof(TaxiTrip.TripDistance)));

    //Console Helper Metotları;
    ConsoleHelper.PeekDataViewInConsole(mlContext, trainingDataView, dataProcessPipeline, 5);
    ConsoleHelper.PeekVectorColumnDataInConsole(mlContext, "Features", trainingDataView, dataProcessPipeline, 5);

    //Model eğitim pipeline'ı oluşturma;
    SdcaRegressionTrainer trainer = mlContext.Regression.Trainers.Sdca(labelColumnName: "Label", featureColumnName: "Features");
    EstimatorChain<RegressionPredictionTransformer<LinearRegressionModelParameters>> trainingPipeline = dataProcessPipeline.Append(trainer);

    //Model eğitimi;
    Console.WriteLine("=============== Training the model ===============");
    TransformerChain<RegressionPredictionTransformer<LinearRegressionModelParameters>> trainedModel = trainingPipeline.Fit(trainingDataView);

    //Model değerlendirilmesi;
    Console.WriteLine("===== Evaluating Model's accuracy with Test data =====");
    IDataView predictions = trainedModel.Transform(testDataView);
    RegressionMetrics metrics = mlContext.Regression.Evaluate(predictions, labelColumnName: "Label", scoreColumnName: "Score");

    //Sonuçların yazdırılması;
    ConsoleHelper.PrintRegressionMetrics(trainer.ToString(), metrics);

    //Modelin kaydedilmesi;
    mlContext.Model.Save(trainedModel, trainingDataView.Schema, ModelPath);
    Console.WriteLine("The model is saved to {0}", ModelPath);
    return trainedModel;//Sonuç döndürülür.
}

void TestSinglePrediction(MLContext mlContext)
{
    //Örnek bir taksi yolculuğu oluşturulur;
    TaxiTrip taxiTripSample = new TaxiTrip()
    {
        VendorId = "VTS",
        RateCode = "1",
        PassengerCount = 1,
        TripTime = 1140,
        TripDistance = 3.75f,
        PaymentType = "CRD",
        FareAmount = 0
    };

    //Eğitilmiş model yüklenir.
    ITransformer trainedModel = mlContext.Model.Load(ModelPath, out DataViewSchema modelInputSchema);
    //Tahmin motoru oluşturulur.
    PredictionEngine<TaxiTrip, TaxiTripFarePrediction> predEngine = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(trainedModel);
    //Örnek üzerinde tahmin yapılır.
    TaxiTripFarePrediction resultprediction = predEngine.Predict(taxiTripSample);
    //Sonuç yazdırılır.
    Console.WriteLine($"**********************************************************************");
    Console.WriteLine($"Predicted fare: {resultprediction.FareAmount:0.####}, actual fare: 15.5");
    Console.WriteLine($"**********************************************************************");
}

//PlotRegressionChart: Metot, MLContext türünde bir nesne (mlContext), bir string (testDataSetPath), bir int (numberOfRecordsToRead) ve bir string dizisi (args) alıyor.
void PlotRegressionChart(MLContext mlContext, string testDataSetPath, int numberOfRecordsToRead, string[] args)
{
    //ITransformer türünde trainedModel adında bir değişken tanımlanıyor.
    ITransformer trainedModel;

    //ModelPath üzerinden FileStream oluşturuluyor. Bu FileStream sadece okuma amacıyla açılıyor ve using bloğu içinde tanımlandığı için kullanım sonunda otomatik olarak kapatılıyor.
    using (FileStream stream = new FileStream(ModelPath, FileMode.Open, FileAccess.Read, FileShare.Read))
    {
        trainedModel = mlContext.Model.Load(stream, out DataViewSchema modelInputSchema);
    }

    //mlContext üzerinden Model.Load metodu çağrılarak eğitilmiş model dosyası (stream) yükleniyor. Yükleme işlemi sonucunda modelInputSchema değişkenine modelin giriş şeması atanıyor.
    PredictionEngine<TaxiTrip, TaxiTripFarePrediction> predFunction = mlContext.Model.CreatePredictionEngine<TaxiTrip, TaxiTripFarePrediction>(trainedModel);

    //Boş bir chartFileName (string türünde) değişkeni tanımlanıyor.
    string chartFileName = "";

    //PLStream türünde pl adında bir nesne oluşturuluyor ve using bloğu içinde tanımlandığı için kullanım sonunda otomatik olarak kapatılacak.
    using (PLStream pl = new PLStream())
    {
        if (args.Length == 1 && args[0] == "svg")//args dizisinin uzunluğu 1 ise ve ilk elemanı "svg" ise,
        {
            pl.sdev("svg");//pl nesnesinin çıktı cihazını "svg" olarak ayarla
            chartFileName = "TaxiRegressionDistribution.svg";//değişkenine "TaxiRegressionDistribution.svg" değeri ata.
            pl.sfnam(chartFileName);//pl.sfnam(chartFileName); satırı, PLplot kütüphanesi tarafından sağlanan bir yöntemdir ve chartFileName değişkeninin değerini kullanarak grafik dosyasının adını belirler. Bu yöntem, grafik çizim işlemlerinde çıktı dosyasının adını ayarlamak için kullanılır. Örneğin, chartFileName değişkeni "TaxiRegressionDistribution.png" veya "TaxiRegressionDistribution.svg" gibi bir değer içeriyorsa, PLplot çıktıyı bu ad ve uzantıya sahip bir dosya olarak oluşturur ve kaydeder.
        }
        else
        {
            pl.sdev("pngcairo");//pl nesnesinin çıktı cihazını "pngcairo" olarak ayarla (pl.sdev("pngcairo")),
            chartFileName = "TaxiRegressionDistribution.png";//chartFileName değişkenine "TaxiRegressionDistribution.png" değeri atanıyor
            pl.sfnam(chartFileName);//pl.sfnam(chartFileName); satırı, PLplot kütüphanesi tarafından sağlanan bir yöntemdir ve chartFileName değişkeninin değerini kullanarak grafik dosyasının adını belirler. Bu yöntem, grafik çizim işlemlerinde çıktı dosyasının adını ayarlamak için kullanılır. Örneğin, chartFileName değişkeni "TaxiRegressionDistribution.png" veya "TaxiRegressionDistribution.svg" gibi bir değer içeriyorsa, PLplot çıktıyı bu ad ve uzantıya sahip bir dosya olarak oluşturur ve kaydeder.
        }

        //pl nesnesinin renk paletini "cmap0_alternate.pal" dosyasından ayarlıyor;
        pl.spal0("cmap0_alternate.pal");

        //pl nesnesini başlatıyor.
        pl.init();

        //xMinLimit, xMaxLimit, yMinLimit, ve yMaxLimit sabitleri belirleniyor. Bunlar grafik sınırları için kullanılacak değerleri temsil eder.
        const int xMinLimit = 0;
        const int xMaxLimit = 35;
        const int yMinLimit = 0;
        const int yMaxLimit = 35;

        //pl nesnesinin çevre ayarlarını (grafik sınırları) belirtilen değerlere göre yapılandırıyor.
        pl.env(xMinLimit, xMaxLimit, yMinLimit, yMaxLimit, AxesScale.Independent, AxisBox.BoxTicksLabelsAxes);

        //Grafikteki karakter ölçeğini ayarlıyor.
        pl.schr(0, 1.25);

        //Grafik için etiketleri ayarlıyor.
        pl.lab("Measured", "Predicted", "Distribution of Taxi Fare Prediction");

        //Grafik renk paletini ayarlıyor.
        pl.col0(1);

        //totalNumber değişkeni, numberOfRecordsToRead parametresinden alınan değeri tutuyor.
        int totalNumber = numberOfRecordsToRead;
        //TaxiTripCsvReader sınıfından bir örnek oluşturulup, GetDataFromCsv metodu kullanılarak testDataSetPath dosyasından totalNumber kadar kayıt okunuyor ve testData listesine dönüştürülüyor.
        List<TaxiTrip> testData = new TaxiTripCsvReader().GetDataFromCsv(testDataSetPath, totalNumber).ToList();

        //code değişkeni, grafikte kullanılacak olan kodu belirliyor.
        char code = (char)9;

        //Grafik rengini ayarlıyor.
        pl.col0(2);

        //Bu dört değişken, grafik çizimi için gereken istatistiksel hesaplamaları yapmak amacıyla tanımlanmıştır. Özellikle, regresyon analizi için eğim (slope) ve kesişim (intercept) hesaplamalarında kullanılırlar.
        double yTotal = 0;//Bu değişken, grafikte gösterilecek olan tahmin edilen (Predicted) fare miktarlarının toplamını tutar. Her iterasyonda, predFunction.Predict(testData[i]).FareAmount ile alınan tahmin edilen fare miktarı yTotal değişkenine eklenir.
        double xTotal = 0;//Bu değişken, grafikte gösterilecek olan ölçülen (Actual) fare miktarlarının toplamını tutar. Her iterasyonda, testData[i].FareAmount ile alınan ölçülen fare miktarı xTotal değişkenine eklenir.
        double xyMultiTotal = 0;//Bu değişken, x ve y değerlerinin çarpımlarının toplamını tutar. Bu toplam, daha sonra regresyon doğrusunun eğimini hesaplamak için kullanılır.
        double xSquareTotal = 0;//Bu değişken, x değerlerinin karelerinin toplamını tutar. Bu toplam, regresyon doğrusunun eğimini hesaplamak için gerekli olan bir diğer terimdir.

        //testData listesinin her bir elemanı üzerinde döngü başlatılıyor.
        for (int i = 0; i < testData.Count; i++)
        {
            double[] x = new double[1];
            double[] y = new double[1];

            //predFunction kullanılarak testData[i] örneği için bir tahmin yapılıyor ve FarePrediction değişkenine atılıyor.
            TaxiTripFarePrediction FarePrediction = predFunction.Predict(testData[i]);

            x[0] = testData[i].FareAmount;//x dizisinin ilk elemanına testData[i] örneğinin FareAmount özelliği atanıyor.
            y[0] = FarePrediction.FareAmount;//y dizisinin ilk elemanına FarePrediction nesnesinin FareAmount özelliği atanıyor.

            //pl nesnesi üzerinden x ve y koordinatlarına göre nokta çiziliyor.
            pl.poin(x, y, code);

            xTotal += x[0];//xTotal, tüm x değerlerinin toplamını tutan değişkene x[0] ekleniyor.
            yTotal += y[0];//yTotal, tüm y değerlerinin toplamını tutan değişkene y[0] ekleniyor.

            double multi = x[0] * y[0];//multi, x[0] ve y[0] değerlerinin çarpımını tutan değişken olarak hesaplanıyor.
            xyMultiTotal += multi;//xyMultiTotal, tüm multi değerlerinin toplamını tutan değişkene multi ekleniyor.

            double xSquare = x[0] * x[0];//xSquare, x[0] değerinin karesini tutan değişken olarak hesaplanıyor.
            xSquareTotal += xSquare;//xSquareTotal, tüm xSquare değerlerinin toplamını tutan değişkene xSquare ekleniyor.

            double ySquare = y[0] * y[0];//Bu değişken, y değerinin karesini hesaplar. Her iterasyonda, y[0] değeri, yani tahmin edilen fare miktarı FarePrediction.FareAmount 'un karesi alınarak ySquare değişkenine atanır. Bu hesaplama, regresyon analizinde çeşitli istatistiksel hesaplamalar için gereklidir. Özetle, ySquare değişkeni, tahmin edilen fare miktarının karesini tutar ve regresyon analizinde önemli bir rol oynar.

            //Konsola Tahmin edilen ve gerçek ücret yazdırılıyor;
            Console.WriteLine($"-------------------------------------------------");
            Console.WriteLine($"Predicted : {FarePrediction.FareAmount}");
            Console.WriteLine($"Actual:    {testData[i].FareAmount}");
            Console.WriteLine($"-------------------------------------------------");
        }

        double minY = yTotal / totalNumber;//minY, tüm y değerlerinin ortalamasını hesaplayarak belirleniyor.
        double minX = xTotal / totalNumber;//minX, tüm x değerlerinin ortalamasını hesaplayarak belirleniyor.
        double minXY = xyMultiTotal / totalNumber;//minXY, tüm xyMultiTotal değerlerinin ortalamasını hesaplayarak belirleniyor.
        double minXsquare = xSquareTotal / totalNumber;//minXsquare, tüm xSquareTotal değerlerinin ortalamasını hesaplayarak belirleniyor.
        double m = ((minX * minY) - minXY) / ((minX * minX) - minXsquare);//m, regresyon çizgisi eğimi (slope) olarak hesaplanıyor.
        double b = minY - (m * minX);//b, regresyon çizgisi yatay ofseti (intercept) olarak hesaplanıyor.
        double x1 = 1;//x1, regresyon çizgisinin ilk noktasının x koordinatı olarak belirleniyor.
        double y1 = (m * x1) + b;//y1, regresyon çizgisinin ilk noktasının y koordinatı olarak hesaplanıyor.
        double x2 = 39;//x2, regresyon çizgisinin ikinci noktasının x koordinatı olarak belirleniyor.
        double y2 = (m * x2) + b;//y2, regresyon çizgisinin ikinci noktasının y koordinatı olarak hesaplanıyor.

        double[] xArray = new double[2];//xArray, regresyon çizgisinin x koordinatlarını tutacak bir dizi olarak oluşturuluyor.
        double[] yArray = new double[2];//yArray, regresyon çizgisinin y koordinatlarını tutacak bir dizi olarak oluşturuluyor.

        xArray[0] = x1;//xArray dizisinin ilk elemanına x1 değeri atanıyor.
        yArray[0] = y1;//yArray dizisinin ilk elemanına y1 değeri atanıyor.
        xArray[1] = x2;//xArray dizisinin ikinci elemanına x2 değeri atanıyor.
        yArray[1] = y2;//yArray dizisinin ikinci elemanına y2 değeri atanıyor.

        pl.col0(4);//Grafik rengini ayarlıyor.
        pl.line(xArray, yArray);//pl nesnesi üzerinden xArray ve yArray dizileriyle bir çizgi çiziyor.
        pl.eop();//pl nesnesini sonlandırıyor.
        pl.gver(out string verText);//PLplot sürüm bilgisini verText değişkenine alıyor.

        //Konsola PLplot sürüm bilgisini yazdırıyor;
        Console.WriteLine("PLplot version " + verText);

    }

    //Konsola "Grafik gösteriliyor..." mesajını yazdırıyor;
    Console.WriteLine("Showing chart...");
    //Process sınıfından p adında bir nesne oluşturuluyor;
    Process p = new Process();
    //chartFileNamePath değişkenine, mevcut dizindeki chartFileName dosyasının yolunu atanıyor.
    string chartFileNamePath = @".\" + chartFileName;
    //p nesnesinin StartInfo özelliğine, chartFileNamePath yolunu içeren yeni bir ProcessStartInfo nesnesi atanıyor.
    p.StartInfo = new ProcessStartInfo(chartFileNamePath)
    {
        UseShellExecute = true
    };
    //p nesnesini başlatıyor, böylece grafik dosyası varsayılan uygulama ile açılıyor;
    p.Start();
}//Bu metod, ML.NET ve PLplot kullanarak bir regresyon analiz sonucunu görselleştirmek için kullanılıyor. Veri setinden alınan değerler üzerinde tahminler yapılıyor ve bu tahminler ile gerçek değerler arasındaki ilişkiyi gösteren bir grafik çiziliyor.

string GetAbsolutePath(string relativePath)//Path metodu.
{
    FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
    string assemblyFolderPath = _dataRoot.Directory.FullName;
    string fullPath = Path.Combine(assemblyFolderPath, relativePath);
    return fullPath;
}

public class TaxiTripCsvReader
{
    public IEnumerable<TaxiTrip> GetDataFromCsv(string dataLocation, int numMaxRecords)
    {
        IEnumerable<TaxiTrip> records = File
            .ReadAllLines(dataLocation)//ReadAllLines: CSV dosyasını satır satır okur.
            .Skip(1)//İlk satırı atlar (varsayılan olarak başlık satırı kabul edilir).
            .Select(x => x.Split(','))//Her satırı virgülle ayırarak diziye dönüştürür.
            .Select(x => new TaxiTrip()//Her dizi öğesi için bir TaxiTrip nesnesi oluşturur ve aşağıdaki şekilde özelliklerini doldurur.
            {
                VendorId = x[0],//Doğrudan dizi öğelerinden alınır.
                RateCode = x[1],//Doğrudan dizi öğelerinden alınır.
                PassengerCount = float.Parse(x[2], CultureInfo.InvariantCulture),//kültür bağımsız bir şekilde float.Parse kullanılarak dönüştürülür.
                TripTime = float.Parse(x[3], CultureInfo.InvariantCulture),//kültür bağımsız bir şekilde float.Parse kullanılarak dönüştürülür.
                TripDistance = float.Parse(x[4], CultureInfo.InvariantCulture),//kültür bağımsız bir şekilde float.Parse kullanılarak dönüştürülür.
                PaymentType = x[5],//Doğrudan dizi öğelerinden alınır.
                FareAmount = float.Parse(x[6], CultureInfo.InvariantCulture)//kültür bağımsız bir şekilde float.Parse kullanılarak dönüştürülür.
            }).Take<TaxiTrip>(numMaxRecords);//Oluşturulan TaxiTrip nesnelerini maksimum kayıt sayısına kadar alır.
        return records;//Oluşturulan TaxiTrip nesnelerinin bir IEnumerable<TaxiTrip> olarak dönüş yapar.
    }
}
