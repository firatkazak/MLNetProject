using Microsoft.ML;
using Octokit;
using GitHubLabeler.DataStructures;
using Microsoft.ML.Data;

internal class Labeler
{
    private readonly GitHubClient _client;//GitHub API'si ile etkileşim kurmak için kullanılır.
    private readonly string _repoOwner;//GitHub deposunun sahibi.
    private readonly string _repoName;//GitHub deposunun adı.
    private readonly string _modelPath;//Eğitimli modelin dosya yolu.
    private readonly MLContext _mlContext;//ML.NET bağlamı.
    private readonly PredictionEngine<GitHubIssue, GitHubIssuePrediction> _predEngine;//Tahmin motoru.
    private readonly ITransformer _trainedModel;//Eğitimli model.
    private FullPrediction[] _fullPredictions;//Tahmin sonuçlarını tutan bir dizi.

    //Yapıcı, sınıfın üyelerini başlatır ve eğitimli modeli dosyadan yükler. Eğer bir erişim token'ı sağlanmışsa, GitHub istemcisi yapılandırılır.
    public Labeler(string modelPath, string repoOwner = "", string repoName = "", string accessToken = "")
    {
        _modelPath = modelPath;
        _repoOwner = repoOwner;
        _repoName = repoName;
        _mlContext = new MLContext();
        _trainedModel = _mlContext.Model.Load(_modelPath, out DataViewSchema modelInputSchema);
        _predEngine = _mlContext.Model.CreatePredictionEngine<GitHubIssue, GitHubIssuePrediction>(_trainedModel);
        if (accessToken != string.Empty)
        {
            ProductHeaderValue productInformation = new ProductHeaderValue("MLGitHubLabeler");
            _client = new GitHubClient(productInformation)
            {
                Credentials = new Credentials(accessToken)
            };
        }
    }

    public void TestPredictionForSingleIssue()//Bu metod, tek bir sabitlenmiş GitHub sorunu için tahmin yapar ve sonuçları ekrana yazdırır.
    {
        GitHubIssue singleIssue = new GitHubIssue()
        {
            ID = "Any-ID",
            Title = "Crash in SqlConnection when using TransactionScope",
            Description = "I'm using SqlClient in netcoreapp2.0. Sqlclient.Close() crashes in Linux but works on Windows"
        };

        GitHubIssuePrediction prediction = _predEngine.Predict(singleIssue);

        _fullPredictions = GetBestThreePredictions(prediction);

        Console.WriteLine($"==== Displaying prediction of Issue with Title = {singleIssue.Title} and Description = {singleIssue.Description} ====");
        Console.WriteLine("1st Label: " + _fullPredictions[0].PredictedLabel + " with score: " + _fullPredictions[0].Score);
        Console.WriteLine("2nd Label: " + _fullPredictions[1].PredictedLabel + " with score: " + _fullPredictions[1].Score);
        Console.WriteLine("3rd Label: " + _fullPredictions[2].PredictedLabel + " with score: " + _fullPredictions[2].Score);
        Console.WriteLine($"=============== Single Prediction - Result: {prediction.Area} ===============");
    }

    private FullPrediction[] GetBestThreePredictions(GitHubIssuePrediction prediction)//Bu metod, en yüksek üç tahmini döndürür.
    {
        float[] scores = prediction.Score;
        int size = scores.Length;
        int index0, index1, index2 = 0;
        VBuffer<ReadOnlyMemory<char>> slotNames = default;
        _predEngine.OutputSchema[nameof(GitHubIssuePrediction.Score)].GetSlotNames(ref slotNames);
        GetIndexesOfTopThreeScores(scores, size, out index0, out index1, out index2);
        _fullPredictions = new FullPrediction[]
            {
                    new FullPrediction(slotNames.GetItemOrDefault(index0).ToString(),scores[index0],index0),
                    new FullPrediction(slotNames.GetItemOrDefault(index1).ToString(),scores[index1],index1),
                    new FullPrediction(slotNames.GetItemOrDefault(index2).ToString(),scores[index2],index2)
            };
        return _fullPredictions;
    }

    private void GetIndexesOfTopThreeScores(float[] scores, int n, out int index0, out int index1, out int index2)//Bu metod, en yüksek üç skora sahip indeksleri belirler.
    {
        int i;
        float first, second, third;
        index0 = index1 = index2 = 0;

        if (n < 3)
        {
            Console.WriteLine("Invalid Input");
            return;
        }

        third = first = second = 000;

        for (i = 0; i < n; i++)
        {
            if (scores[i] > first)
            {
                third = second;
                second = first;
                first = scores[i];
            }
            else if (scores[i] > second)
            {
                third = second;
                second = scores[i];
            }
            else if (scores[i] > third)
                third = scores[i];
        }

        List<float> scoresList = scores.ToList();
        index0 = scoresList.IndexOf(first);
        index1 = scoresList.IndexOf(second);
        index2 = scoresList.IndexOf(third);
    }

    public async Task LabelAllNewIssuesInGitHubRepo()//Bu metod, GitHub reposundaki yeni sorunları etiketler.
    {
        IReadOnlyList<Issue> newIssues = await GetNewIssues();
        foreach (Issue issue in newIssues.Where(issue => !issue.Labels.Any()))
        {
            FullPrediction[] label = PredictLabels(issue);
            ApplyLabels(issue, label);
        }
    }

    private async Task<IReadOnlyList<Issue>> GetNewIssues()//Bu metod, GitHub reposundaki yeni sorunları alır.
    {
        RepositoryIssueRequest issueRequest = new RepositoryIssueRequest
        {
            State = ItemStateFilter.Open,
            Filter = IssueFilter.All,
            Since = DateTime.Now.AddMinutes(-10)
        };
        IReadOnlyList<Issue> allIssues = await _client.Issue.GetAllForRepository(_repoOwner, _repoName, issueRequest);
        return allIssues.Where(i => !i.HtmlUrl.Contains("/pull/")).ToList();
    }

    private FullPrediction[] PredictLabels(Issue issue)//Bu metod, bir sorunun etiketlerini tahmin eder.
    {
        GitHubIssue corefxIssue = new GitHubIssue
        {
            ID = issue.Number.ToString(),
            Title = issue.Title,
            Description = issue.Body
        };
        _fullPredictions = Predict(corefxIssue);
        return _fullPredictions;
    }

    public FullPrediction[] Predict(GitHubIssue issue)//Bu metod, bir GitHub sorunu için tahmin yapar.
    {
        GitHubIssuePrediction prediction = _predEngine.Predict(issue);
        FullPrediction[] fullPredictions = GetBestThreePredictions(prediction);
        return fullPredictions;
    }

    private void ApplyLabels(Issue issue, FullPrediction[] fullPredictions)//
    {
        IssueUpdate issueUpdate = new IssueUpdate();
        foreach (FullPrediction fullPrediction in fullPredictions)
        {
            if (fullPrediction.Score >= 0.3)
            {
                issueUpdate.AddLabel(fullPrediction.PredictedLabel);
                _client.Issue.Update(_repoOwner, _repoName, issue.Number, issueUpdate);
                Console.WriteLine($"Issue {issue.Number} : \"{issue.Title}\" \t was labeled as: {fullPredictions[0].PredictedLabel}");
            }
        }
    }
}
