using Microsoft.Extensions.Configuration;

namespace GitHubLabeler;
public class Extensions
{
    public Extensions(IConfiguration configuration, MyTrainerStrategy myTrainerStrategy)
    {
        Configuration = configuration;
        MyTrainerStrategy = myTrainerStrategy;
    }

    public static IConfiguration Configuration { get; set; }
    //Configuration, uygulamanızın yapılandırma ayarlarını içeren bir yapılandırma nesnesidir.
    public static MyTrainerStrategy MyTrainerStrategy { get; set; }
    //MyTrainerStrategy, hangi makine öğrenimi algoritmasının kullanılacağını belirten bir numaralandırmadır.İki seçenek vardır:
    //SdcaMultiClassTrainer: Stochastic Dual Coordinate Ascent algoritması
    //OVAAveragedPerceptronTrainer: One-vs-All Averaged Perceptron algoritması
}

public enum MyTrainerStrategy : int
{
    SdcaMultiClassTrainer = 1,
    OVAAveragedPerceptronTrainer = 2
};
//Bu sınıf, uygulamanızın yapılandırma ayarlarını (Configuration) ve kullanılan eğitim stratejisini (MyTrainerStrategy) depolamak için kullanılır. Extensions sınıfı, IConfiguration ve MyTrainerStrategy parametreleri ile başlatılır.
