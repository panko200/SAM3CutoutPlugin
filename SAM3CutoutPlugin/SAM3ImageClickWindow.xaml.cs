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

public partial class SAM3ImageClickWindow : Window
{
    // [X, Y, Label (1=Positive, 0=Negative)]
    public List<Tuple<int, int, int>> Points { get; private set; } = new();
    
    // [X1, Y1, X2, Y2]
    public List<Tuple<int, int, int, int>> Boxes { get; private set; } = new();

    public bool IsConfirmed { get; private set; } = false;

    public string TextPrompt => PromptTextBox.Text;
    public string TextMode => MatchBestRadioButton.IsChecked == true ? "best" : "all";

    private readonly BitmapSource _imageSource;
    private readonly string _inputPath;
    private readonly string _outputDir;
    private readonly string _token;
    private readonly string _outputFormatStr;
    private readonly string _scriptPath;

    // ボックスドラッグ描画用ステート
    private Point _startDragPoint;
    private bool _isDraggingBox = false;
    private Rectangle? _dragPreviewRect;
    private readonly List<Rectangle> _drawnBoxShapes = new();
    private bool _isUpdatingMode = false;

    public SAM3ImageClickWindow(BitmapSource imageSource, string inputPath, string outputDir, string token, string outputFormatStr, string scriptPath)
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

    private Point ConstrainToImage(Point p)
    {
        double x = Math.Max(0, Math.Min(_imageSource.PixelWidth, p.X));
        double y = Math.Max(0, Math.Min(_imageSource.PixelHeight, p.Y));
        return new Point(x, y);
    }

    private void Image_MouseLeftButtonDown(object sender, MouseButtonEventArgs e)
    {
        if (BoxModeRadioButton.IsChecked == true)
        {
            var pos = e.GetPosition(PreviewImage);
            _startDragPoint = ConstrainToImage(pos);
            _isDraggingBox = true;
            ImageContainer.CaptureMouse();

            _dragPreviewRect = new Rectangle
            {
                Stroke = Brushes.Aqua,
                StrokeThickness = 2,
                Fill = new SolidColorBrush(Color.FromArgb(0x30, 0x00, 0xFF, 0xFF))
            };
            Canvas.SetLeft(_dragPreviewRect, _startDragPoint.X);
            Canvas.SetTop(_dragPreviewRect, _startDragPoint.Y);
            _dragPreviewRect.Width = 0;
            _dragPreviewRect.Height = 0;
            MarkerCanvas.Children.Add(_dragPreviewRect);
        }
        else
        {
            AddPoint(e.GetPosition(PreviewImage), isPositive: true);
        }
    }

    private void Image_MouseRightButtonDown(object sender, MouseButtonEventArgs e)
    {
        if (BoxModeRadioButton.IsChecked == true)
        {
            return;
        }
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

    private void ImageContainer_MouseMove(object sender, MouseEventArgs e)
    {
        if (_isDraggingBox && _dragPreviewRect != null)
        {
            var pos = e.GetPosition(PreviewImage);
            var currentPoint = ConstrainToImage(pos);

            double left = Math.Min(_startDragPoint.X, currentPoint.X);
            double top = Math.Min(_startDragPoint.Y, currentPoint.Y);
            double width = Math.Abs(_startDragPoint.X - currentPoint.X);
            double height = Math.Abs(_startDragPoint.Y - currentPoint.Y);

            Canvas.SetLeft(_dragPreviewRect, left);
            Canvas.SetTop(_dragPreviewRect, top);
            _dragPreviewRect.Width = width;
            _dragPreviewRect.Height = height;
        }
    }

    private void ImageContainer_MouseLeftButtonUp(object sender, MouseButtonEventArgs e)
    {
        if (_isDraggingBox)
        {
            _isDraggingBox = false;
            ImageContainer.ReleaseMouseCapture();

            if (_dragPreviewRect != null)
            {
                var pos = e.GetPosition(PreviewImage);
                var currentPoint = ConstrainToImage(pos);

                double left = Math.Min(_startDragPoint.X, currentPoint.X);
                double top = Math.Min(_startDragPoint.Y, currentPoint.Y);
                double width = Math.Abs(_startDragPoint.X - currentPoint.X);
                double height = Math.Abs(_startDragPoint.Y - currentPoint.Y);

                if (width > 5 && height > 5)
                {
                    // 有効なボックスとして登録
                    int x1 = (int)Math.Round(left);
                    int y1 = (int)Math.Round(top);
                    int x2 = (int)Math.Round(left + width);
                    int y2 = (int)Math.Round(top + height);

                    Boxes.Add(new Tuple<int, int, int, int>(x1, y1, x2, y2));
                    _drawnBoxShapes.Add(_dragPreviewRect);
                    _dragPreviewRect = null;
                }
                else
                {
                    // 小さすぎる場合は破棄
                    MarkerCanvas.Children.Remove(_dragPreviewRect);
                    _dragPreviewRect = null;
                }
            }
        }
    }

    private void ClearInputs()
    {
        Points?.Clear();
        Boxes?.Clear();
        _drawnBoxShapes?.Clear();
        if (MarkerCanvas != null)
        {
            MarkerCanvas.Children.Clear();
        }
        if (MaskOverlayImage != null)
        {
            MaskOverlayImage.Source = null;
        }
        if (PreviewStatusText != null)
        {
            PreviewStatusText.Text = "";
        }
    }

    private void ClearButton_Click(object sender, RoutedEventArgs e)
    {
        ClearInputs();
    }

    private void Mode_Checked(object sender, RoutedEventArgs e)
    {
        if (_isUpdatingMode) return;
        if (PointModeRadioButton == null || BoxModeRadioButton == null || PromptTextBox == null) return;
        _isUpdatingMode = true;
        try
        {
            ClearInputs();
            if (PointModeRadioButton.IsChecked == true)
            {
                PromptTextBox.Text = "";
            }
        }
        finally
        {
            _isUpdatingMode = false;
        }
    }

    private void PromptTextBox_TextChanged(object sender, TextChangedEventArgs e)
    {
        if (_isUpdatingMode) return;
        if (PointModeRadioButton == null || BoxModeRadioButton == null || PromptTextBox == null) return;
        string text = PromptTextBox.Text.Trim();
        if (!string.IsNullOrEmpty(text))
        {
            if (PointModeRadioButton.IsChecked == true)
            {
                _isUpdatingMode = true;
                try
                {
                    ClearInputs();
                    BoxModeRadioButton.IsChecked = true;
                }
                finally
                {
                    _isUpdatingMode = false;
                }
            }
        }
    }

    private async void PreviewButton_Click(object sender, RoutedEventArgs e)
    {
        string textPrompt = PromptTextBox.Text.Trim();
        bool hasInput = false;
        if (PointModeRadioButton.IsChecked == true)
        {
            hasInput = Points.Count > 0;
        }
        else // Box mode
        {
            hasInput = Boxes.Count > 0 || !string.IsNullOrWhiteSpace(textPrompt);
        }

        if (!hasInput)
        {
            MessageBox.Show("クリックでポイント/ボックスを指定するか、テキストプロンプトを入力してください。", "プレビューエラー", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        PreviewButton.IsEnabled = false;
        PreviewStatusText.Text = "SAM3で推論中...";
        MaskOverlayImage.Source = null;

        try
        {
            var cmdObj = new {
                action = "preview",
                points = PointModeRadioButton.IsChecked == true ? Points.Select(p => new int[] { p.Item1, p.Item2 }).ToArray() : new int[0][],
                labels = PointModeRadioButton.IsChecked == true ? Points.Select(p => p.Item3).ToArray() : new int[0],
                boxes = BoxModeRadioButton.IsChecked == true ? Boxes.Select(b => new int[] { b.Item1, b.Item2, b.Item3, b.Item4 }).ToArray() : new int[0][],
                output_dir = _outputDir,
                text_prompt = textPrompt,
                text_mode = TextMode
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
        string textPrompt = PromptTextBox.Text.Trim();
        bool hasInput = false;
        if (PointModeRadioButton.IsChecked == true)
        {
            hasInput = Points.Count > 0;
        }
        else // Box mode
        {
            hasInput = Boxes.Count > 0 || !string.IsNullOrWhiteSpace(textPrompt);
        }

        if (!hasInput)
        {
            MessageBox.Show("切り抜く対象をクリック/ドラッグするか、テキストプロンプトを入力してください。", "エラー", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        IsConfirmed = true;
        Close();
    }
}
