using MovieRecommender.Services;
using Microsoft.Extensions.ML;
using MovieRecommender.DataStructures;

var builder = WebApplication.CreateBuilder(args);

builder.Configuration.AddJsonFile("appsettings.json", optional: false, reloadOnChange: true);
//Bu kod, yapýlandýrma (configuration) kaynaðý olarak appsettings.json dosyasýný ekler.
//AddJsonFile("appsettings.json", optional: false, reloadOnChange: true):
//optional: false: Bu, appsettings.json dosyasýnýn bulunmasýnýn zorunlu olduðunu belirtir. Dosya bulunamazsa uygulama baþlatýlamaz.
//reloadOnChange: true: Bu, appsettings.json dosyasýnda deðiþiklik yapýldýðýnda yapýlandýrmanýn otomatik olarak yeniden yükleneceðini belirtir.

builder.Services.AddControllersWithViews();
builder.Services.AddRazorPages();
builder.Services.AddSingleton<IProfileService, ProfileService>();
builder.Services.AddSingleton<IMovieService, MovieService>();
builder.Services.AddPredictionEnginePool<MovieRating, MovieRatingPrediction>().FromFile(builder.Configuration["MLModelPath"]);
//PredictionEnginePool, ML.NET modelini kullanarak tahminler yapabilen bir havuzdur.
//Bu havuz, performansý artýrmak için PredictionEngine örneklerini yeniden kullanýr.
//AddPredictionEnginePool<MovieRating, MovieRatingPrediction>(): MovieRating giriþ tipi ve MovieRatingPrediction çýkýþ tipi ile bir PredictionEnginePool oluþturur.
//.FromFile(builder.Configuration["MLModelPath"]): Model dosyasýnýn yolunu appsettings.json dosyasýndan okur ve bu model dosyasýný kullanarak PredictionEngine havuzunu yapýlandýrýr.

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
