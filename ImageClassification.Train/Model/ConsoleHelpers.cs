namespace ImageClassification.Train.Model;

public static class ConsoleHelpers
{
    //ConsoleWriteHeader: Konsol ekranına başlık yazdırmak için kullanılır. Başlıklar sarı renkte ve '#' karakterleriyle çevrelenmiş olarak yazdırılır.
    public static void ConsoleWriteHeader(params string[] lines)
    {
        ConsoleColor defaultColor = Console.ForegroundColor;
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine(" ");
        foreach (string line in lines)
        {
            Console.WriteLine(line);
        }
        int maxLength = lines.Select(x => x.Length).Max();
        Console.WriteLine(new String('#', maxLength));
        Console.ForegroundColor = defaultColor;
    }

    //ConsolePressAnyKey: Kullanıcıdan herhangi bir tuşa basması için bekleme yapar. Konsol ekranına "Press any key to finish." mesajını yeşil renkte yazdırır ve bir tuşa basılmasını bekler.
    public static void ConsolePressAnyKey()
    {
        ConsoleColor defaultColor = Console.ForegroundColor;
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine(" ");
        Console.WriteLine("Press any key to finish.");
        Console.ReadKey();
    }

    //ConsoleWriteException: Hata durumlarında konsola hata mesajlarını yazdırmak için kullanılır. Hata başlığı ("EXCEPTION") kırmızı renkte ve '#' karakterleriyle çevrelenmiş olarak yazdırılır. Hata mesajları kırmızı renkte yazdırılır.
    public static void ConsoleWriteException(params string[] lines)
    {
        ConsoleColor defaultColor = Console.ForegroundColor;
        Console.ForegroundColor = ConsoleColor.Red;
        const string exceptionTitle = "EXCEPTION";
        Console.WriteLine(" ");
        Console.WriteLine(exceptionTitle);
        Console.WriteLine(new String('#', exceptionTitle.Length));
        Console.ForegroundColor = defaultColor;
        foreach (string line in lines)
        {
            Console.WriteLine(line);
        }
    }

    //ConsoleWriteImagePrediction: Bir görüntünün sınıflandırma tahmin sonuçlarını konsola yazdırmak için kullanılır.
    //ImagePath: Görüntünün dosya yolunu temsil eder.
    //Label: Görüntünün gerçek etiketini temsil eder.
    //PredictedLabel: Model tarafından tahmin edilen etiket değerini temsil eder.
    //Probability: Tahmin edilen sınıfın olasılık skorunu temsil eder.
    public static void ConsoleWriteImagePrediction(string ImagePath, string Label, string PredictedLabel, float Probability)
    {
        ConsoleColor defaultForeground = Console.ForegroundColor;
        ConsoleColor labelColor = ConsoleColor.Magenta;
        ConsoleColor probColor = ConsoleColor.Blue;

        Console.Write("ImagePath: ");
        Console.ForegroundColor = labelColor;
        Console.Write($"{Path.GetFileName(ImagePath)}");
        Console.ForegroundColor = defaultForeground;
        Console.Write(" original labeled as ");
        Console.ForegroundColor = labelColor;
        Console.Write(Label);
        Console.ForegroundColor = defaultForeground;
        Console.Write(" predicted as ");
        Console.ForegroundColor = labelColor;
        Console.Write(PredictedLabel);
        Console.ForegroundColor = defaultForeground;
        Console.Write(" with score ");
        Console.ForegroundColor = probColor;
        Console.Write(Probability);
        Console.ForegroundColor = defaultForeground;
        Console.WriteLine("");
    }
}
