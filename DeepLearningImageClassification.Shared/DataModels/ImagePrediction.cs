﻿using Microsoft.ML.Data;

namespace DeepLearningImageClassification.Shared.DataModels;
public class ImagePrediction
{
    [ColumnName("Score")]
    public float[] Score;

    [ColumnName("PredictedLabel")]
    public string PredictedLabel;
}
