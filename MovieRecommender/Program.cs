using MovieRecommender.Services;
using Microsoft.Extensions.ML;
using MovieRecommender.DataStructures;

var builder = WebApplication.CreateBuilder(args);

builder.Configuration.AddJsonFile("appsettings.json", optional: false, reloadOnChange: true);
//Bu kod, yapılandırma (configuration) kaynağı olarak appsettings.json dosyasını ekler.
//AddJsonFile("appsettings.json", optional: false, reloadOnChange: true):
//optional: false: Bu, appsettings.json dosyasının bulunmasının zorunlu olduğunu belirtir. Dosya bulunamazsa uygulama başlatılamaz.
//reloadOnChange: true: Bu, appsettings.json dosyasında değişiklik yapıldığında yapılandırmanın otomatik olarak yeniden yükleneceğini belirtir.

builder.Services.AddControllersWithViews();
builder.Services.AddRazorPages();
builder.Services.AddSingleton<IProfileService, ProfileService>();
builder.Services.AddSingleton<IMovieService, MovieService>();
builder.Services.AddPredictionEnginePool<MovieRating, MovieRatingPrediction>().FromFile(builder.Configuration["MLModelPath"]);
//PredictionEnginePool, ML.NET modelini kullanarak tahminler yapabilen bir havuzdur.
//Bu havuz, performansı artırmak için PredictionEngine örneklerini yeniden kullanır.
//AddPredictionEnginePool<MovieRating, MovieRatingPrediction>(): MovieRating giriş tipi ve MovieRatingPrediction çıkış tipi ile bir PredictionEnginePool oluşturur.
//.FromFile(builder.Configuration["MLModelPath"]): Model dosyasının yolunu appsettings.json dosyasından okur ve bu model dosyasını kullanarak PredictionEngine havuzunu yapılandırır.

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
