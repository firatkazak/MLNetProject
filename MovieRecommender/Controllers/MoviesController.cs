using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.ML;
using Microsoft.ML;
using MovieRecommender.DataStructures;
using MovieRecommender.Models;
using MovieRecommender.Services;
using Newtonsoft.Json;
using System.Text;

namespace MovieRecommender.Controllers;
public class MoviesController : Controller
{
    private readonly IMovieService _movieService;
    private readonly IProfileService _profileService;
    private readonly PredictionEnginePool<MovieRating, MovieRatingPrediction> _model;

    public MoviesController(IMovieService movieService, IProfileService profileService, PredictionEnginePool<MovieRating, MovieRatingPrediction> model)
    {
        _movieService = movieService;
        _profileService = profileService;
        _model = model;
    }

    public IActionResult Choose()
    {
        return View(_movieService.GetSomeSuggestions());
    }

    public IActionResult Recommend(int id)
    {
        Profile activeprofile = _profileService.GetProfileByID(id);
        MLContext mlContext = new MLContext();
        List<(int movieId, float normalizedScore)> ratings = new List<(int movieId, float normalizedScore)>();
        List<(int movieId, int movieRating)> MovieRatings = _profileService.GetProfileWatchedMovies(id) ?? new List<(int movieId, int movieRating)>();
        List<Movie> WatchedMovies = new List<Movie>();

        foreach ((int movieId, int movieRating) in MovieRatings)
        {
            WatchedMovies.Add(_movieService.Get(movieId));
        }

        MovieRatingPrediction prediction = null;

        foreach (Movie movie in _movieService.GetTrendingMovies)
        {
            prediction = _model.Predict(new MovieRating
            {
                userId = id.ToString(),
                movieId = movie.MovieID.ToString()
            });
            float normalizedscore = Sigmoid(prediction.Score);
            ratings.Add((movie.MovieID, normalizedscore));
        }

        ViewData["watchedmovies"] = WatchedMovies;
        ViewData["ratings"] = ratings;
        ViewData["trendingmovies"] = _movieService.GetTrendingMovies;
        return View(activeprofile);
    }

    public IActionResult Watched(int id)
    {
        Profile activeprofile = _profileService.GetProfileByID(id);
        List<(int movieId, int movieRating)> MovieRatings = _profileService.GetProfileWatchedMovies(id) ?? new List<(int movieId, int movieRating)>();
        List<Movie> WatchedMovies = new List<Movie>();

        foreach ((int movieId, float normalizedScore) in MovieRatings)
        {
            WatchedMovies.Add(_movieService.Get(movieId));
        }

        ViewData["watchedmovies"] = WatchedMovies;
        ViewData["trendingmovies"] = _movieService.GetTrendingMovies;
        return View(activeprofile);
    }


    public IActionResult Watch()
    {
        return View();
    }

    public IActionResult Profiles()
    {
        List<Profile> profiles = _profileService.GetProfiles;
        return View(profiles);
    }


    public float Sigmoid(float x)
    {
        return (float)(100 / (1 + Math.Exp(-x)));
    }

    public class JsonContent : StringContent
    {
        public JsonContent(object obj) : base(JsonConvert.SerializeObject(obj), Encoding.UTF8, "application/json") { }
    }
}
