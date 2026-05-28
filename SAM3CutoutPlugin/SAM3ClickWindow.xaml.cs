using System;
using System.Collections.Generic;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace SAM3CutoutPlugin;

public partial class SAM3ClickWindow : Window
{
    // [X, Y, Label (1=Positive, 0=Negative)]
    public List<Tuple<int, int, int>> Points { get; private set; } = new();
    public bool IsConfirmed { get; private set; } = false;

    public string TextPrompt => "";

    private readonly BitmapSource _imageSource;
    private readonly string _inputPath;
    private readonly string _outputDir;
    private readonly string _token;
    private readonly string _outputFormatStr;
    private readonly string _scriptPath;

    public SAM3ClickWindow(BitmapSource imageSource, string inputPath, string outputDir, string token, string outputFormatStr, string scriptPath)
    {
        InitializeComponent();
        _imageSource = imageSource;
        _inputPath = inputPath;
        _outputDir = outputDir;
        _token = token;
        _outputFormatStr = outputFormatStr;
        _scriptPath = scriptPath;
        PreviewImage.Source = _imageSource;
        
        // ImageContainerを元の画像サイズに合わせることで座標計算を正確にする
        ImageContainer.Width = _imageSource.PixelWidth;
        ImageContainer.Height = _imageSource.PixelHeight;
    }

    private void Image_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
    {
        AddPoint(e.GetPosition(PreviewImage), isPositive: true);
    }

    private void Image_MouseRightButtonDown(object sender, MouseButtonEventArgs e)
    {
        AddPoint(e.GetPosition(PreviewImage), isPositive: false);
    }

    private void AddPoint(Point pos, bool isPositive)
    {
        // 座標を整数化
        int x = (int)Math.Round(pos.X);
        int y = (int)Math.Round(pos.Y);

        Points.Add(new Tuple<int, int, int>(x, y, isPositive ? 1 : 0));

        // マーカーを描画
        var ellipse = new Ellipse
        {
            Width = 10,
            Height = 10,
            Fill = isPositive ? Brushes.Aqua : Brushes.Red,
            Stroke = Brushes.White,
            StrokeThickness = 2
        };

        Canvas.SetLeft(ellipse, pos.X - 5);
        Canvas.SetTop(ellipse, pos.Y - 5);
        MarkerCanvas.Children.Add(ellipse);
    }

    private void ClearButton_Click(object sender, RoutedEventArgs e)
    {
        Points.Clear();
        MarkerCanvas.Children.Clear();
        MaskOverlayImage.Source = null;
        PreviewStatusText.Text = "";
    }

    private async void PreviewButton_Click(object sender, RoutedEventArgs e)
    {
        if (Points.Count == 0)
        {
            MessageBox.Show("クリックでポイントを指定してください。", "プレビューエラー", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        PreviewButton.IsEnabled = false;
        PreviewStatusText.Text = "SAM3で推論中...";
        MaskOverlayImage.Source = null;

        try
        {
            var cmdObj = new {
                action = "preview",
                points = Points.Select(p => new int[] { p.Item1, p.Item2 }).ToArray(),
                labels = Points.Select(p => p.Item3).ToArray(),
                output_dir = _outputDir
            };
            string jsonCmd = System.Text.Json.JsonSerializer.Serialize(cmdObj);

            string maskPath = await PythonEnvManager.SendCommandAsync(jsonCmd, null, System.Threading.CancellationToken.None);

            if (!string.IsNullOrEmpty(maskPath) && System.IO.File.Exists(maskPath))
            {
                var img = new BitmapImage();
                img.BeginInit();
                img.CacheOption = BitmapCacheOption.OnLoad;
                img.CreateOptions = BitmapCreateOptions.IgnoreImageCache;
                img.UriSource = new Uri(maskPath);
                img.EndInit();
                img.Freeze();
                MaskOverlayImage.Source = img;
                PreviewStatusText.Text = "更新完了";
            }
            else
            {
                PreviewStatusText.Text = "マスクが生成されませんでした";
            }
        }
        catch (Exception ex)
        {
            PreviewStatusText.Text = "エラー発生";
            MessageBox.Show($"プレビュー生成中にエラーが発生しました:\n{ex.Message}", "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
        }
        finally
        {
            PreviewButton.IsEnabled = true;
        }
    }

    private void RunButton_Click(object sender, RoutedEventArgs e)
    {
        if (Points.Count == 0)
        {
            MessageBox.Show("切り抜く対象をクリックしてください。", "エラー", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        IsConfirmed = true;
        Close();
    }
}
