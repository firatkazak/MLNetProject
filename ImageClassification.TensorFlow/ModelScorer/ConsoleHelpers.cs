using ImageClassification.TensorFlow.ImageDataStructures;

namespace ImageClassification.TensorFlow.ModelScorer;
public static class ConsoleHelpers
{
    //ConsoleWriteHeader: Verilen parametrelerden oluşan başlık satırlarını sarı renkte konsola yazar.
    //ConsoleWriteHeader: Başlık satırlarının üzerine başlık uzunluğunda '#' işaretleriyle bir çizgi çizer.
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

    //ConsolePressAnyKey: Yeşil renkte "Press any key to finish." mesajını konsola yazar ve kullanıcı herhangi bir tuşa basana kadar bekler.
    public static void ConsolePressAnyKey()
    {
        ConsoleColor defaultColor = Console.ForegroundColor;
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine(" ");
        Console.WriteLine("Press any key to finish.");
        Console.ForegroundColor = defaultColor;
        Console.ReadKey();
    }

    //Verilen parametrelerden oluşan hata mesajını konsola kırmızı renkte yazarak başlık olarak "EXCEPTION" ve "#" işaretleri ekler.
    public static void ConsoleWriteException(params string[] lines)
    {
        ConsoleColor defaultColor = Console.ForegroundColor;
        const string exceptionTitle = "EXCEPTION";

        Console.WriteLine(" ");
        Console.ForegroundColor = ConsoleColor.Red;
        Console.WriteLine(exceptionTitle);
        Console.WriteLine(new String('#', exceptionTitle.Length));
        Console.ForegroundColor = defaultColor;
        foreach (string line in lines)
        {
            Console.WriteLine(line);
        }
    }

    //ImageNetDataProbability nesnesini konsola yazdırır. ImagePath özelliğini dosya adı olarak magenta(bordo) renkte yazar. Gerçek etiketi (Label) ve tahmin edilen etiketi (PredictedLabel) farklıysa tahmin rengini kırmızıya, aynıysa yeşile boyar. Olasılığı (Probability) mavi renkte yazdırır.
    public static void ConsoleWrite(this ImageNetDataProbability self)
    {
        ConsoleColor defaultForeground = Console.ForegroundColor;
        ConsoleColor labelColor = ConsoleColor.Magenta;
        ConsoleColor probColor = ConsoleColor.Blue;
        ConsoleColor exactLabel = ConsoleColor.Green;
        ConsoleColor failLabel = ConsoleColor.Red;

        Console.Write("ImagePath: ");
        Console.ForegroundColor = labelColor;
        Console.Write($"{Path.GetFileName(self.ImagePath)}");
        Console.ForegroundColor = defaultForeground;
        Console.Write(" labeled as ");
        Console.ForegroundColor = labelColor;
        Console.Write(self.Label);
        Console.ForegroundColor = defaultForeground;
        Console.Write(" predicted as ");
        if (self.Label.Equals(self.PredictedLabel))
        {
            Console.ForegroundColor = exactLabel;
            Console.Write($"{self.PredictedLabel}");
        }
        else
        {
            Console.ForegroundColor = failLabel;
            Console.Write($"{self.PredictedLabel}");
        }
        Console.ForegroundColor = defaultForeground;
        Console.Write(" with probability ");
        Console.ForegroundColor = probColor;
        Console.Write(self.Probability);
        Console.ForegroundColor = defaultForeground;
        Console.WriteLine("");
    }
}
