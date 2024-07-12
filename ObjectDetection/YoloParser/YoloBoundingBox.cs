using System.Drawing;

namespace ObjectDetection.YoloParser;
public class BoundingBoxDimensions : DimensionsBase { }//`BoundingBoxDimensions : DimensionsBase` ifadesi, bir sınıfın başka bir sınıftan türediğini (inheritance) gösterir. Yani `BoundingBoxDimensions` sınıfı, `DimensionsBase` sınıfından türetilmiştir veya ondan miras almıştır. Bu durumda `BoundingBoxDimensions` sınıfı, `DimensionsBase` sınıfının tüm özelliklerini (`X`, `Y`, `Height`, `Width`) içerir ve ayrıca kendi ekstra özelliklerini veya davranışlarını tanımlayabilir. Miras alma (inheritance), bir sınıfın diğer bir sınıfın özelliklerini ve davranışlarını yeniden kullanmasını sağlar, böylece kod tekrarını azaltır ve kod organizasyonunu geliştirir. Yani kısacası, `BoundingBoxDimensions : DimensionsBase` ifadesi, `BoundingBoxDimensions` sınıfının `DimensionsBase` sınıfından türediğini veya ondan miras aldığını belirtir.

//YoloBoundingBox: YOLO modeli tarafından tespit edilen bir nesnenin bilgilerini içerir.
public class YoloBoundingBox
{
    public BoundingBoxDimensions Dimensions { get; set; }//BoundingBoxDimensions tipinde bir boyut nesnesi tutar.

    public string Label { get; set; }//Nesnenin sınıf etiketini (örneğin, kedi, köpek, araba) belirtir.

    public float Confidence { get; set; }//Nesnenin tespit edilme olasılığını ifade eder.

    //Rect: Nesnenin sınırlayıcı dikdörtgenini (RectangleF olarak) döndürür.
    public RectangleF Rect
    {
        get { return new RectangleF(Dimensions.X, Dimensions.Y, Dimensions.Width, Dimensions.Height); }
    }

    public Color BoxColor { get; set; }// Nesnenin sınırlayıcı kutusunun renk bilgisini tutar.
}
