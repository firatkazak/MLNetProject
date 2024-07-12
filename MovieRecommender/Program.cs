using MovieRecommender.Services;
using Microsoft.Extensions.ML;
using MovieRecommender.DataStructures;

var builder = WebApplication.CreateBuilder(args);

builder.Configuration.AddJsonFile("appsettings.json", optional: false, reloadOnChange: true);
//Bu kod, yap�land�rma (configuration) kayna�� olarak appsettings.json dosyas�n� ekler.
//AddJsonFile("appsettings.json", optional: false, reloadOnChange: true):
//optional: false: Bu, appsettings.json dosyas�n�n bulunmas�n�n zorunlu oldu�unu belirtir. Dosya bulunamazsa uygulama ba�lat�lamaz.
//reloadOnChange: true: Bu, appsettings.json dosyas�nda de�i�iklik yap�ld���nda yap�land�rman�n otomatik olarak yeniden y�klenece�ini belirtir.

builder.Services.AddControllersWithViews();
builder.Services.AddRazorPages();
builder.Services.AddSingleton<IProfileService, ProfileService>();
builder.Services.AddSingleton<IMovieService, MovieService>();
builder.Services.AddPredictionEnginePool<MovieRating, MovieRatingPrediction>().FromFile(builder.Configuration["MLModelPath"]);
//PredictionEnginePool, ML.NET modelini kullanarak tahminler yapabilen bir havuzdur.
//Bu havuz, performans� art�rmak i�in PredictionEngine �rneklerini yeniden kullan�r.
//AddPredictionEnginePool<MovieRating, MovieRatingPrediction>(): MovieRating giri� tipi ve MovieRatingPrediction ��k�� tipi ile bir PredictionEnginePool olu�turur.
//.FromFile(builder.Configuration["MLModelPath"]): Model dosyas�n�n yolunu appsettings.json dosyas�ndan okur ve bu model dosyas�n� kullanarak PredictionEngine havuzunu yap�land�r�r.

var app = builder.Build();

if (!app.Environment.IsDevelopment())
{
    app.UseExceptionHandler("/Home/Error");
    app.UseHsts();
}

app.UseHttpsRedirection();
app.UseStaticFiles();
app.UseRouting();
app.UseAuthorization();

app.MapControllerRoute(name: "default", pattern: "{controller=Movies}/{action=Profiles}/{id?}");

app.Run();
