namespace ImageClassification.Train.ImageData;

public class ImagePrediction
{
    public float[] Score;//Tahmin edilen sonuçlar için olasılık skorlarını içeren bir float dizisi.
    public string PredictedLabelValue;//Tahmin edilen etiket değerini içeren bir string. Model tarafından tahmin edilen çiçek türü gibi.
}

//ImageWithLabelPrediction: ImagePrediction sınıfını genişleterek bir etiket alanı ekler.
public class ImageWithLabelPrediction : ImagePrediction
{
    public ImageWithLabelPrediction(ImagePrediction pred, string label)
    {
        Label = label;
        Score = pred.Score;
        PredictedLabelValue = pred.PredictedLabelValue;
    }
    public string Label;
}
