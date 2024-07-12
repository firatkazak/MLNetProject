namespace MovieRecommender;

public class Extension
{
    public Extension(IConfiguration configuration)
    {
        Configuration = configuration;
    }

    public static IConfiguration Configuration { get; set; }
}
