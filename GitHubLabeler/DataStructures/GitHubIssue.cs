using Microsoft.ML.Data;

namespace GitHubLabeler.DataStructures;
internal class GitHubIssue
{
    [LoadColumn(0)]
    public string ID;//Sorunun ID'si.

    [LoadColumn(1)]
    public string Area;//Sorunun etiketi, örneğin "area-System.Threading" gibi.

    [LoadColumn(2)]
    public string Title;//Sorunun başlığı.

    [LoadColumn(3)]
    public string Description;//Sorunun açıklaması.
}
//GitHubIssue sınıfı, bir GitHub sorununun özelliklerini temsil eder.