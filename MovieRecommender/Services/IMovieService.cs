using MovieRecommender.Models;

namespace MovieRecommender.Services;

public interface IMovieService
{
    Movie Get(int id);
    IEnumerable<Movie> GetAllMovies();
    string GetModelPath();
    IEnumerable<Movie> GetRecentMovies();
    IEnumerable<Movie> GetSomeSuggestions();

    List<Movie> GetTrendingMovies { get; }
}
