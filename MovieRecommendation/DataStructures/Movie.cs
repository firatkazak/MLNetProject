namespace MovieRecommendation.DataStructures;
class Movie//Filmlerin temsil edildiği bir sınıftır.
{
    public int movieId;//Film Id'si.
    public string movieTitle;//Film başlığı.
    static string moviesdatasetRelativepath = @"C:\Users\firat\source\repos\MLDersleri\MovieRecommendation\Data";
    static string moviesdatasetpath = @"C:\Users\firat\source\repos\MLDersleri\MovieRecommendation\Data\recommendation-movies.csv";
    public Lazy<List<Movie>> _movies = new Lazy<List<Movie>>(() => LoadMovieData(moviesdatasetpath));
    //Lazy yüklemesi kullanarak, filmlerin yalnızca ihtiyaç duyulduğunda yüklenmesini sağlar.
    public Movie Get(int id)//Id'ye göre getiriyor.
    {
        return _movies.Value.Single(m => m.movieId == id);
    }

    //LoadMovieData: recommendation-movies.csv dosyasından film verilerini yükler.
    private static List<Movie> LoadMovieData(String moviesdatasetpath)
    {
        List<Movie> result = new List<Movie>();
        Stream fileReader = File.OpenRead(moviesdatasetpath);
        StreamReader reader = new StreamReader(fileReader);
        try
        {
            bool header = true;
            int index = 0;
            string line = "";
            while (!reader.EndOfStream)
            {
                if (header)
                {
                    line = reader.ReadLine();
                    header = false;
                }
                line = reader.ReadLine();
                string[] fields = line.Split(',');
                int movieId = Int32.Parse(fields[0].ToString().TrimStart(new char[] { '0' }));
                string movieTitle = fields[1].ToString();
                result.Add(new Movie() { movieId = movieId, movieTitle = movieTitle });
                index++;
            }
        }
        finally
        {
            if (reader != null)
            {
                reader.Dispose();
            }
        }
        return result;
    }
}
