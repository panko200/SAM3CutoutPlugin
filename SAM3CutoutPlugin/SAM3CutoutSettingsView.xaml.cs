using System.Windows.Controls;

namespace SAM3CutoutPlugin;

public partial class SAM3CutoutSettingsView : UserControl
{
    public SAM3CutoutSettingsView()
    {
        InitializeComponent();
    }

    private async void SetupButton_Click(object sender, System.Windows.RoutedEventArgs e)
    {
        SetupButton.IsEnabled = false;
        SetupProgressText.Text = "構築を開始します...";
        SetupProgressBar.Value = 0;

        var progress = new System.Progress<(string message, double pct)>(p =>
        {
            SetupProgressText.Text = p.message;
            SetupProgressBar.Value = p.pct;
        });

        try
        {
            await PythonEnvManager.SetupEnvironmentAsync(progress, System.Threading.CancellationToken.None);
            SetupProgressText.Text = "構築完了！ (トークンを設定して閉じてください)";
            SetupProgressText.Foreground = System.Windows.Media.Brushes.Green;
        }
        catch (System.Exception ex)
        {
            SetupProgressText.Text = $"エラー: {ex.Message}";
            SetupProgressText.Foreground = System.Windows.Media.Brushes.Red;
        }
        finally
        {
            SetupButton.IsEnabled = true;
        }
    }
}
