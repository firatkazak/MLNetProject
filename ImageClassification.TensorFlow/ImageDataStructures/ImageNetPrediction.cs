using ImageClassification.TensorFlow.ModelScorer;
using Microsoft.ML.Data;

namespace ImageClassification.TensorFlow.ImageDataStructures;

//Bu sınıf, TensorFlow modelinizin tahmin ettiği etiketleri (labels) temsil eder. ColumnName niteliği, bu özelliğin hangi sütundan alındığını belirtir.
public class ImageNetPrediction
{
    [ColumnName(TFModelScorer.InceptionSettings.outputTensorName)]
    public float[] PredictedLabels;
}
