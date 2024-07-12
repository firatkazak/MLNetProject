using System.Drawing;
using System.Drawing.Drawing2D;
using ObjectDetection.YoloParser;
using ObjectDetection.DataStructures;
using ObjectDetection;
using Microsoft.ML;

//Yollar
string assetsRelativePath = @"C:\Users\firat\source\repos\MLDersleri\ObjectDetection\Assets";
string assetsPath = GetAbsolutePath(assetsRelativePath);
string modelFilePath = Path.Combine(assetsPath, "Model", "TinyYolo2_model.onnx");
string imagesFolder = Path.Combine(assetsPath, "Images");
string outputFolder = Path.Combine(assetsPath, "Images", "Output");

// context nesnesi.
MLContext mlContext = new MLContext();

try
{
    //ImageNetData sınıfı üzerinden görüntü verilerini okuyarak bir koleksiyon oluşturur;
    IEnumerable<ImageNetData> images = ImageNetData.ReadFromFile(imagesFolder);

    //mlContext.Data.LoadFromEnumerable ile veriler IDataView türüne yüklenir;
    IDataView imageDataView = mlContext.Data.LoadFromEnumerable(images);

    //OnnxModelScorer sınıfı oluşturularak ONNX modelinin yükleme ve değerlendirme işlemleri için kullanılır;
    OnnxModelScorer modelScorer = new OnnxModelScorer(imagesFolder, modelFilePath, mlContext);

    //modelScorer.Score metodu ile görüntü verileri üzerinde nesne tespiti yapılır ve sonuç olasılıkları elde edilir.
    IEnumerable<float[]> probabilities = modelScorer.Score(imageDataView);

    //YoloOutputParser sınıfı ile her bir olasılık dizisi için nesne tespiti çıktıları ayrıştırılır ve filtrelenir.
    YoloOutputParser parser = new YoloOutputParser();

    //modelin her bir görüntü için döndürdüğü olasılık dizilerinin koleksiyonudur.
    //Her bir olasılık dizisi, bir görüntüdeki her bir hücre için nesne tespiti olasılıklarını içerir.
    IEnumerable<IList<YoloBoundingBox>> boundingBoxes =
        probabilities.Select(probability => parser.ParseOutputs(probability)).Select(boxes => parser.FilterBoundingBoxes(boxes, 5, .5F));

    for (int i = 0; i < images.Count(); i++)
    {
        //Bu satır, i numaralı görüntünün dosya adını alır ve imageFileName değişkenine atar.
        string imageFileName = images.ElementAt(i).Label;
        //Bu satır, i numaralı görüntüde tespit edilen nesne kutularını alır ve detectedObjects değişkenine atar.
        IList<YoloBoundingBox> detectedObjects = boundingBoxes.ElementAt(i);
        //Aşağıda tanımladığımız metot;
        DrawBoundingBox(imagesFolder, outputFolder, imageFileName, detectedObjects);
        //LogDetectedObjects fonksiyonu ile tespit edilen nesneler konsola yazdırılır.
        LogDetectedObjects(imageFileName, detectedObjects);
    }
}
catch (Exception ex)
{
    Console.WriteLine(ex.ToString());
}

Console.WriteLine("========= End of Process..Hit any Key ========");

//Yol metodu;
string GetAbsolutePath(string relativePath)
{
    FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
    string assemblyFolderPath = _dataRoot.Directory.FullName;
    string fullPath = Path.Combine(assemblyFolderPath, relativePath);
    return fullPath;
}

//Tespit edilen nesneler konsola yazdırılır;
void LogDetectedObjects(string imageName, IList<YoloBoundingBox> boundingBoxes)
{
    Console.WriteLine($".....The objects in the image {imageName} are detected as below....");

    foreach (YoloBoundingBox box in boundingBoxes)
    {
        Console.WriteLine($"{box.Label} and its Confidence score: {box.Confidence}");
    }

    Console.WriteLine("");
}

//DrawBoundingBox: Her bir tespit edilen nesne için etiketlenmiş kutular ve olasılık yüzdesi ile metinler çizilir.
void DrawBoundingBox(string inputImageLocation, string outputImageLocation, string imageName, IList<YoloBoundingBox> filteredBoundingBoxes)
{
    Image image = Image.FromFile(Path.Combine(inputImageLocation, imageName));

    int originalImageHeight = image.Height;
    int originalImageWidth = image.Width;

    foreach (YoloBoundingBox box in filteredBoundingBoxes)
    {
        uint x = (uint)Math.Max(box.Dimensions.X, 0);
        uint y = (uint)Math.Max(box.Dimensions.Y, 0);
        uint width = (uint)Math.Min(originalImageWidth - x, box.Dimensions.Width);
        uint height = (uint)Math.Min(originalImageHeight - y, box.Dimensions.Height);

        x = (uint)originalImageWidth * x / OnnxModelScorer.ImageNetSettings.imageWidth;
        y = (uint)originalImageHeight * y / OnnxModelScorer.ImageNetSettings.imageHeight;
        width = (uint)originalImageWidth * width / OnnxModelScorer.ImageNetSettings.imageWidth;
        height = (uint)originalImageHeight * height / OnnxModelScorer.ImageNetSettings.imageHeight;

        string text = $"{box.Label} ({(box.Confidence * 100).ToString("0")}%)";

        using (Graphics thumbnailGraphic = Graphics.FromImage(image))
        {
            thumbnailGraphic.CompositingQuality = CompositingQuality.HighQuality;
            thumbnailGraphic.SmoothingMode = SmoothingMode.HighQuality;
            thumbnailGraphic.InterpolationMode = InterpolationMode.HighQualityBicubic;

            Font drawFont = new Font("Arial", 12, FontStyle.Bold);
            SizeF size = thumbnailGraphic.MeasureString(text, drawFont);
            SolidBrush fontBrush = new SolidBrush(Color.Black);
            Point atPoint = new Point((int)x, (int)y - (int)size.Height - 1);

            Pen pen = new Pen(box.BoxColor, 3.2f);
            SolidBrush colorBrush = new SolidBrush(box.BoxColor);

            thumbnailGraphic.FillRectangle(colorBrush, (int)x, (int)(y - size.Height - 1), (int)size.Width, (int)size.Height);
            thumbnailGraphic.DrawString(text, drawFont, fontBrush, atPoint);

            thumbnailGraphic.DrawRectangle(pen, x, y, width, height);
        }
    }

    if (!Directory.Exists(outputImageLocation))
    {
        Directory.CreateDirectory(outputImageLocation);
    }

    image.Save(Path.Combine(outputImageLocation, imageName));
}
