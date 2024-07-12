using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;

string TrainingDataLocation = @"C:\Users\firat\source\repos\MLDersleri\ProductRecommender\Data\Amazon0302.txt";//Verinin konumu.

MLContext mlContext = new MLContext();//Context nesnesi.

//mlContext.Data.LoadFromTextFile metoduyla traindata adında bir IDataView yüklenir. Bu, Amazon0302.txt dosyasından verileri yükler ve sütunları belirtir.
IDataView traindata = mlContext.Data.LoadFromTextFile(path: TrainingDataLocation, columns: new[]
{
    new TextLoader.Column("Label", DataKind.Single, 0),
    new TextLoader.Column(name:nameof(ProductEntry.ProductID), dataKind:DataKind.UInt32, source: new [] { new TextLoader.Range(0) }, keyCount: new KeyCount(262111)),
    new TextLoader.Column(name:nameof(ProductEntry.CoPurchaseProductID), dataKind:DataKind.UInt32, source: new [] { new TextLoader.Range(1) }, keyCount: new KeyCount(262111))
}, hasHeader: true, separatorChar: '\t');

//MatrixFactorizationTrainer.Options sınıfından bir options nesnesi oluşturulur ve Matrix Factorization algoritması için gerekli ayarlar burada yapılır.
MatrixFactorizationTrainer.Options options = new MatrixFactorizationTrainer.Options();
options.MatrixColumnIndexColumnName = nameof(ProductEntry.ProductID);//MatrixColumnIndexColumnName özelliği, modelin matrisindeki sütunların isimlerini belirtir. Burada ProductID özelliği, ürünlerin matris sütun indeksi olarak kullanılacaktır.
options.MatrixRowIndexColumnName = nameof(ProductEntry.CoPurchaseProductID);//MatrixRowIndexColumnName özelliği, modelin matrisindeki satırların isimlerini belirtir. CoPurchaseProductID özelliği, ürünlerin matris satır indeksi olarak kullanılacaktır.
options.LabelColumnName = "Label";//LabelColumnName özelliği, eğitim veri setindeki etiketleri içeren sütunun adını belirtir. Burada "Label" adında bir sütun, her bir eğitim örneğinin etiketini (label) taşır.
options.LossFunction = MatrixFactorizationTrainer.LossFunctionType.SquareLossOneClass;//LossFunction özelliği, modelin eğitim sırasında kullanacağı kayıp fonksiyonunu belirtir.
//SquareLossOneClass, karesel kayıp (square loss) fonksiyonunu kullanarak eğitim yapılacağını belirtir.
options.Alpha = 0.01;//Alpha özelliği, modelin öğrenme hızını (learning rate) belirler. Daha yüksek Alpha değerleri, modelin daha hızlı öğrenmesini sağlar.
options.Lambda = 0.025;//Lambda özelliği, modelin regülarizasyon parametresidir. Bu parametre, modelin aşırı uyumu (overfitting) önlemek için kullanılır. Daha yüksek Lambda değerleri, daha fazla regülarizasyon sağlar.

//mlContext.Recommendation().Trainers.MatrixFactorization(options) çağrısıyla MatrixFactorizationTrainer nesnesi est oluşturulur.
MatrixFactorizationTrainer est = mlContext.Recommendation().Trainers.MatrixFactorization(options);

//est.Fit(traindata) metoduyla model eğitimi yapılır. Bu adımda traindata kullanılarak Matrix Factorization modeli eğitilir.
ITransformer model = est.Fit(traindata);

//mlContext.Model.CreatePredictionEngine metoduyla bir prediction engine (predictionengine) oluşturulur. Bu engine, eğitilen modeli kullanarak tahmin yapmak için kullanılacak.
PredictionEngine<ProductEntry, Copurchase_prediction> predictionengine = mlContext.Model.CreatePredictionEngine<ProductEntry, Copurchase_prediction>(model);

//predictionengine.Predict metoduyla belirtilen ProductID ve CoPurchaseProductID değerlerine göre bir tahmin yapılır.
Copurchase_prediction prediction = predictionengine.Predict(new ProductEntry()
{
    ProductID = 3,
    CoPurchaseProductID = 63
});

//Tahmin edilen skoru Console.WriteLine ile yazdırılır ve kullanıcıdan herhangi bir tuşa basıncı beklenir;
Console.WriteLine("\n For ProductID = 3 and  CoPurchaseProductID = 63 the predicted score is " + Math.Round(prediction.Score, 1));
Console.WriteLine("=============== End of process, hit any key to finish ===============");
Console.ReadKey();

public class Copurchase_prediction//Tahmin sınıfı.
{
    public float Score { get; set; }
}

public class ProductEntry//Model eğitimi sınıfı.
{
    [KeyType(count: 262111)]
    public uint ProductID { get; set; }

    [KeyType(count: 262111)]
    public uint CoPurchaseProductID { get; set; }
}

//Çıktı açıklaması: Verilen girdiye göre (`ProductID = 3` ve `CoPurchaseProductID = 63`), model tarafından tahmin edilen skor `0.4` olarak belirlenmiştir. Bu skor, bu iki ürünün birlikte alınma olasılığını temsil eder. Matrix Factorization yöntemi, veri setindeki ürünler arasındaki ilişkileri analiz ederek bu tür tahminlerde bulunur. Tahmin skoru ne kadar yüksekse, o kadar yüksek bir olasılıkla belirtilen ürünlerin birlikte satın alınacağı öngörülmüş olur. Bu sonuç, Amazon0302.txt gibi bir veri dosyasından eğitilen ve `MatrixFactorizationTrainer` kullanılarak oluşturulan bir modelin tahmin yeteneğini gösterir.