using System.Drawing;

namespace ObjectDetection.YoloParser;
class YoloOutputParser
{
    class CellDimensions : DimensionsBase { }// Hücre boyutları için bir alt sınıfı temsil eder.

    //// YOLO model parametreleri ve sabitleri
    public const int ROW_COUNT = 13;
    public const int COL_COUNT = 13;
    public const int CHANNEL_COUNT = 125;
    public const int BOXES_PER_CELL = 5;
    public const int BOX_INFO_FEATURE_COUNT = 5;
    public const int CLASS_COUNT = 20;
    public const float CELL_WIDTH = 32;
    public const float CELL_HEIGHT = 32;

    private int channelStride = ROW_COUNT * COL_COUNT;

    // YOLO modeli için kullanılan ölçekler ve sınıf etiketleri;

    private float[] anchors = new float[]
    {
            1.08F, 1.19F, 3.42F, 4.41F, 6.63F, 11.38F, 9.42F, 5.11F, 16.62F, 10.52F
    };

    private string[] labels = new string[]
    {
            "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person",
            "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    };

    private static Color[] classColors = new Color[]
    {
            Color.Khaki,
            Color.Fuchsia,
            Color.Silver,
            Color.RoyalBlue,
            Color.Green,
            Color.DarkOrange,
            Color.Purple,
            Color.Gold,
            Color.Red,
            Color.Aquamarine,
            Color.Lime,
            Color.AliceBlue,
            Color.Sienna,
            Color.Orchid,
            Color.Tan,
            Color.LightPink,
            Color.Yellow,
            Color.HotPink,
            Color.OliveDrab,
            Color.SandyBrown,
            Color.DarkTurquoise
    };

    //// Sigmoid(): YOLO modeli çıktılarını işlemek için sigmoid ve softmax gibi aktivasyon fonksiyonları kullanılır.
    private float Sigmoid(float value)
    {
        float k = (float)Math.Exp(value);
        return k / (1.0f + k);
    }

    //Softmax(): YOLO modeli çıktılarını işlemek için sigmoid ve softmax gibi aktivasyon fonksiyonları kullanılır.
    private float[] Softmax(float[] values)
    {
        float maxVal = values.Max();
        IEnumerable<double> exp = values.Select(v => Math.Exp(v - maxVal));
        double sumExp = exp.Sum();
        return exp.Select(v => (float)(v / sumExp)).ToArray();
    }

    //GetOffset(): Model çıktılarından belirli bir hücrenin ofsetini hesaplar;
    private int GetOffset(int x, int y, int channel)
    {
        return (channel * this.channelStride) + (y * COL_COUNT) + x;
    }

    //ExtractBoundingBoxDimensions(): Model çıktılarından sınırlayıcı kutu boyutlarını çıkarır;
    private BoundingBoxDimensions ExtractBoundingBoxDimensions(float[] modelOutput, int x, int y, int channel)
    {
        return new BoundingBoxDimensions
        {
            X = modelOutput[GetOffset(x, y, channel)],
            Y = modelOutput[GetOffset(x, y, channel + 1)],
            Width = modelOutput[GetOffset(x, y, channel + 2)],
            Height = modelOutput[GetOffset(x, y, channel + 3)]
        };
    }

    //GetConfidence(): Model çıktılarından güven puanını çıkarır;
    private float GetConfidence(float[] modelOutput, int x, int y, int channel)
    {
        return Sigmoid(modelOutput[GetOffset(x, y, channel + 4)]);
    }

    //MapBoundingBoxToCell(): Sınırlayıcı kutuyu hücre boyutlarına eşler;
    private CellDimensions MapBoundingBoxToCell(int x, int y, int box, BoundingBoxDimensions boxDimensions)
    {
        return new CellDimensions
        {
            X = ((float)x + Sigmoid(boxDimensions.X)) * CELL_WIDTH,
            Y = ((float)y + Sigmoid(boxDimensions.Y)) * CELL_HEIGHT,
            Width = (float)Math.Exp(boxDimensions.Width) * CELL_WIDTH * anchors[box * 2],
            Height = (float)Math.Exp(boxDimensions.Height) * CELL_HEIGHT * anchors[box * 2 + 1],
        };
    }

    //ExtractClasses(): Model çıktılarından sınıf tahminlerini çıkarır;
    public float[] ExtractClasses(float[] modelOutput, int x, int y, int channel)
    {
        float[] predictedClasses = new float[CLASS_COUNT];
        int predictedClassOffset = channel + BOX_INFO_FEATURE_COUNT;
        for (int predictedClass = 0; predictedClass < CLASS_COUNT; predictedClass++)
        {
            predictedClasses[predictedClass] = modelOutput[GetOffset(x, y, predictedClass + predictedClassOffset)];
        }
        return Softmax(predictedClasses);
    }

    //GetTopResult(): Tahmin edilen sınıflar arasından en yüksek puanı bulur;
    private ValueTuple<int, float> GetTopResult(float[] predictedClasses)
    {
        return predictedClasses
            .Select((predictedClass, index) => (Index: index, Value: predictedClass))
            .OrderByDescending(result => result.Value)
            .First();
    }

    //IntersectionOverUnion(): İki sınırlayıcı kutu arasındaki Intersection over Union (IoU) hesaplar;
    private float IntersectionOverUnion(RectangleF boundingBoxA, RectangleF boundingBoxB)
    {
        float areaA = boundingBoxA.Width * boundingBoxA.Height;

        if (areaA <= 0)
            return 0;

        float areaB = boundingBoxB.Width * boundingBoxB.Height;

        if (areaB <= 0)
            return 0;

        float minX = Math.Max(boundingBoxA.Left, boundingBoxB.Left);
        float minY = Math.Max(boundingBoxA.Top, boundingBoxB.Top);
        float maxX = Math.Min(boundingBoxA.Right, boundingBoxB.Right);
        float maxY = Math.Min(boundingBoxA.Bottom, boundingBoxB.Bottom);

        float intersectionArea = Math.Max(maxY - minY, 0) * Math.Max(maxX - minX, 0);

        return intersectionArea / (areaA + areaB - intersectionArea);
    }

    //ParseOutputs(): Model çıktılarını analiz ederek tespit edilen nesneleri listeler;
    public IList<YoloBoundingBox> ParseOutputs(float[] yoloModelOutputs, float threshold = .3F)
    {
        List<YoloBoundingBox> boxes = new List<YoloBoundingBox>();

        for (int row = 0; row < ROW_COUNT; row++)
        {
            for (int column = 0; column < COL_COUNT; column++)
            {
                for (int box = 0; box < BOXES_PER_CELL; box++)
                {
                    int channel = (box * (CLASS_COUNT + BOX_INFO_FEATURE_COUNT));

                    BoundingBoxDimensions boundingBoxDimensions = ExtractBoundingBoxDimensions(yoloModelOutputs, row, column, channel);

                    float confidence = GetConfidence(yoloModelOutputs, row, column, channel);

                    CellDimensions mappedBoundingBox = MapBoundingBoxToCell(row, column, box, boundingBoxDimensions);

                    if (confidence < threshold)
                        continue;

                    float[] predictedClasses = ExtractClasses(yoloModelOutputs, row, column, channel);

                    var (topResultIndex, topResultScore) = GetTopResult(predictedClasses);
                    float topScore = topResultScore * confidence;

                    if (topScore < threshold)
                        continue;

                    boxes.Add(new YoloBoundingBox()
                    {
                        Dimensions = new BoundingBoxDimensions
                        {
                            X = (mappedBoundingBox.X - mappedBoundingBox.Width / 2),
                            Y = (mappedBoundingBox.Y - mappedBoundingBox.Height / 2),
                            Width = mappedBoundingBox.Width,
                            Height = mappedBoundingBox.Height,
                        },
                        Confidence = topScore,
                        Label = labels[topResultIndex],
                        BoxColor = classColors[topResultIndex]
                    });
                }
            }
        }
        return boxes;
    }

    //FilterBoundingBoxes(): Tespit edilen nesneler arasında filtreleme yapar; (confidence ve IoU temelinde)
    public IList<YoloBoundingBox> FilterBoundingBoxes(IList<YoloBoundingBox> boxes, int limit, float threshold)
    {
        int activeCount = boxes.Count;
        bool[] isActiveBoxes = new bool[boxes.Count];

        for (int i = 0; i < isActiveBoxes.Length; i++)
            isActiveBoxes[i] = true;

        var sortedBoxes = boxes
            .Select((b, i) => new { Box = b, Index = i })
            .OrderByDescending(b => b.Box.Confidence)
            .ToList();

        List<YoloBoundingBox> results = new List<YoloBoundingBox>();

        for (int i = 0; i < boxes.Count; i++)
        {
            if (isActiveBoxes[i])
            {
                YoloBoundingBox boxA = sortedBoxes[i].Box;
                results.Add(boxA);

                if (results.Count >= limit)
                    break;

                for (int j = i + 1; j < boxes.Count; j++)
                {
                    if (isActiveBoxes[j])
                    {
                        YoloBoundingBox boxB = sortedBoxes[j].Box;

                        if (IntersectionOverUnion(boxA.Rect, boxB.Rect) > threshold)
                        {
                            isActiveBoxes[j] = false;
                            activeCount--;

                            if (activeCount <= 0)
                                break;
                        }
                    }
                }

                if (activeCount <= 0)
                    break;
            }
        }
        return results;
    }
}
