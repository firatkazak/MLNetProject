namespace ImageClassification.Train.ImageData;

public class ImagePredictionEx
{
    public string ImagePath;//Resmin dosya yolunu temsil eden bir string.
    public string Label;//Resmin gerçek etiketini temsil eden bir string.
    public string PredictedLabelValue;//Model tarafından tahmin edilen etiket değerini içeren bir string.
    public float[] Score;//Tahmin edilen sonuçlar için olasılık skorlarını içeren bir float dizisi.
}
