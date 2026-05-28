using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;

namespace SAM3CutoutPlugin;

public class TargetObject
{
    public int Id { get; set; }

    // ★ 表示名：後から名前が自由に入力できるように、カスタムネームプロパティを用意
    public string CustomName { get; set; } = "";
    public string Name => string.IsNullOrEmpty(CustomName) ? $"オブジェクト {Id}" : CustomName;

    // ★ 修正：ToStringを定義しておくことで、ComboBoxにクラス名ではなく「オブジェクト 1」と自動表示されます
    public override string ToString() => Name;

    // [X, Y, Label (1=Positive, 0=Negative)]
    public List<Tuple<int, int, int>> Points { get; } = new();

    // [X1, Y1, X2, Y2]
    public List<Tuple<int, int, int, int>> Boxes { get; } = new();
}

public partial class SAM3VideoClickWindow : Window
{
    public List<TargetObject> TargetObjects { get; private set; } = new();
    private TargetObject _currentObject;
    private bool _isInitializingObjects = false;

    public bool IsConfirmed { get; private set; } = false;

    public string TextPrompt => PromptTextBox.Text;
    public string TextMode => MatchBestRadioButton.IsChecked == true ? "best" : "all";

    // プロンプトを打った基準フレームをViewModelへ渡す
    public int SelectedFrameIndex => _currentPreviewFrame;

    private readonly BitmapSource _imageSource;
    private readonly string _inputPath;
    private readonly string _outputDir;
    private readonly string _token;
    private readonly string _outputFormatStr;
    private readonly string _scriptPath;
    private int _currentPreviewFrame = 0;

    // ボックスドラッグ用
    private Point _startDragPoint;
    private bool _isDraggingBox = false;
    private Rectangle? _dragPreviewRect;
    private readonly List<Rectangle> _drawnBoxShapes = new();
    private bool _isUpdatingMode = false;

    // スライダーシークのキャンセル用
    private CancellationTokenSource? _sliderCts;

    // マスクおよびポイントを描画する「6色周期カラーテーブル」
    private static readonly Color[] ObjectColors = new Color[]
    {
        Color.FromRgb(0, 100, 255),    // 1: 青
        Color.FromRgb(255, 0, 0),      // 2: 赤
        Color.FromRgb(0, 200, 0),      // 3: 緑
        Color.FromRgb(255, 200, 0),    // 4: 黄
        Color.FromRgb(255, 0, 255),    // 5: 紫
        Color.FromRgb(0, 220, 220)     // 6: 水
    };

    public SAM3VideoClickWindow(BitmapSource imageSource, string inputPath, string outputDir, string token, string outputFormatStr, string scriptPath)
    {
        InitializeComponent();
        _imageSource = imageSource;
        _inputPath = inputPath;
        _outputDir = outputDir;
        _token = token;
        _outputFormatStr = outputFormatStr;
        _scriptPath = scriptPath;
        PreviewImage.Source = _imageSource;

        ImageContainer.Width = _imageSource.PixelWidth;
        ImageContainer.Height = _imageSource.PixelHeight;

        // 総フレーム数からスライダーを調整
        int totalFrames = PythonEnvManager.TotalFrames;
        if (totalFrames > 0)
        {
            FrameSlider.Maximum = totalFrames - 1;
            FrameIndexText.Text = $"0 / {totalFrames - 1}";
        }
        else
        {
            FrameSlider.IsEnabled = false;
        }

        // オブジェクトの初期設定 (デフォルトはオブジェクト1)
        _isInitializingObjects = true;
        var defaultObj = new TargetObject { Id = 1 };
        TargetObjects.Add(defaultObj);
        ObjectComboBox.ItemsSource = TargetObjects;
        ObjectComboBox.SelectedItem = defaultObj;
        _currentObject = defaultObj;
        _isInitializingObjects = false;
    }

    private Brush GetBrushForObject(int objectId, bool isPositive)
    {
        if (!isPositive) return Brushes.Red; // 除外ポイント(右クリック)は常に赤

        int index = (objectId - 1) % ObjectColors.Length;
        return new SolidColorBrush(ObjectColors[index]);
    }

    // メモリ上のBase64バイト配列から直接、超高速かつ完全に非ロックでBitmapImageをビルドする
    private static BitmapImage LoadBitmapFromBase64(string b64)
    {
        byte[] bytes = Convert.FromBase64String(b64);
        var ms = new MemoryStream(bytes);
        var bitmap = new BitmapImage();
        bitmap.BeginInit();
        bitmap.CacheOption = BitmapCacheOption.OnLoad;
        bitmap.StreamSource = ms;
        bitmap.EndInit();
        bitmap.Freeze();
        return bitmap;
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
        if (BoxModeRadioButton.IsChecked == true) return;
        AddPoint(e.GetPosition(PreviewImage), isPositive: false);
    }

    private void AddPoint(Point pos, bool isPositive)
    {
        if (_currentObject == null) return;

        int x = (int)Math.Round(pos.X);
        int y = (int)Math.Round(pos.Y);

        _currentObject.Points.Add(new Tuple<int, int, int>(x, y, isPositive ? 1 : 0));
        RedrawMarkers();
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
                    int x1 = (int)Math.Round(left);
                    int y1 = (int)Math.Round(top);
                    int x2 = (int)Math.Round(left + width);
                    int y2 = (int)Math.Round(top + height);

                    _currentObject.Boxes.Add(new Tuple<int, int, int, int>(x1, y1, x2, y2));
                    RedrawMarkers();
                    _dragPreviewRect = null;
                }
                else
                {
                    MarkerCanvas.Children.Remove(_dragPreviewRect);
                    _dragPreviewRect = null;
                }
            }
        }
    }

    private void ClearButton_Click(object sender, RoutedEventArgs e) => ClearInputs();

    // スライダー移動時の処理
    private async void FrameSlider_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
    {
        if (FrameIndexText == null) return;
        _currentPreviewFrame = (int)FrameSlider.Value;
        FrameIndexText.Text = $"{_currentPreviewFrame} / {(int)FrameSlider.Maximum}";

        _sliderCts?.Cancel();
        _sliderCts = new CancellationTokenSource();
        var ct = _sliderCts.Token;

        try
        {
            await Task.Delay(50, ct);
            await UpdateFrameOnlyAsync(ct);
        }
        catch (TaskCanceledException) { }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Slider error: {ex.Message}");
        }
    }

    private async Task UpdateFrameOnlyAsync(CancellationToken ct)
    {
        try
        {
            var cmdObj = new
            {
                action = "get_frame",
                frame_index = _currentPreviewFrame,
                output_dir = _outputDir
            };
            string jsonCmd = System.Text.Json.JsonSerializer.Serialize(cmdObj);

            string responseJson = await PythonEnvManager.SendCommandAsync(jsonCmd, null, ct);

            if (string.IsNullOrEmpty(responseJson)) return;

            using (var doc = System.Text.Json.JsonDocument.Parse(responseJson))
            {
                var root = doc.RootElement;
                string b64Frame = root.GetProperty("frame").GetString() ?? "";
                string b64Mask = root.GetProperty("mask").GetString() ?? "";

                Dispatcher.Invoke(() =>
                {
                    if (!string.IsNullOrEmpty(b64Frame))
                    {
                        PreviewImage.Source = LoadBitmapFromBase64(b64Frame);
                    }

                    if (b64Mask == "NONE")
                    {
                        MaskOverlayImage.Source = null;
                        PreviewStatusText.Text = $"フレーム {_currentPreviewFrame} 表示中 (プレビュー未生成)";
                    }
                    else if (!string.IsNullOrEmpty(b64Mask))
                    {
                        MaskOverlayImage.Source = LoadBitmapFromBase64(b64Mask);
                        PreviewStatusText.Text = $"フレーム {_currentPreviewFrame} 表示中 (プレビューあり)";
                    }
                });
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Frame update error: {ex.Message}");
        }
    }

    private async void PreviewButton_Click(object sender, RoutedEventArgs e)
    {
        PreviewButton.IsEnabled = false;
        try
        {
            await UpdatePreviewAsync(CancellationToken.None);
        }
        finally
        {
            PreviewButton.IsEnabled = true;
        }
    }

    private async Task UpdatePreviewAsync(CancellationToken ct)
    {
        string textPrompt = PromptTextBox.Text.Trim();
        bool hasInput = TargetObjects.Any(o => o.Points.Count > 0 || o.Boxes.Count > 0) || !string.IsNullOrWhiteSpace(textPrompt);

        if (!hasInput) return;

        PreviewStatusText.Text = "SAM3で推論中...";

        try
        {
            var cmdObj = new
            {
                action = "preview",
                objects = GetObjectsData(),
                output_dir = _outputDir,
                text_prompt = textPrompt,
                text_mode = TextMode,
                frame_index = _currentPreviewFrame
            };
            string jsonCmd = System.Text.Json.JsonSerializer.Serialize(cmdObj);

            string responseJson = await PythonEnvManager.SendCommandAsync(jsonCmd, null, ct);

            if (string.IsNullOrEmpty(responseJson)) return;

            using (var doc = System.Text.Json.JsonDocument.Parse(responseJson))
            {
                var root = doc.RootElement;
                string b64Frame = root.GetProperty("frame").GetString() ?? "";
                string b64Mask = root.GetProperty("mask").GetString() ?? "";

                Dispatcher.Invoke(() =>
                {
                    if (!string.IsNullOrEmpty(b64Frame))
                    {
                        PreviewImage.Source = LoadBitmapFromBase64(b64Frame);
                    }
                    if (!string.IsNullOrEmpty(b64Mask))
                    {
                        MaskOverlayImage.Source = LoadBitmapFromBase64(b64Mask);
                    }
                    PreviewStatusText.Text = $"フレーム {_currentPreviewFrame} プレビュー完了";
                });
            }
        }
        catch (OperationCanceledException) { }
        catch (Exception ex)
        {
            Dispatcher.Invoke(() => {
                PreviewStatusText.Text = "エラー発生";
            });
            System.Diagnostics.Debug.WriteLine($"Preview error: {ex.Message}");
        }
    }

    private void RunButton_Click(object sender, RoutedEventArgs e)
    {
        string textPrompt = PromptTextBox.Text.Trim();
        bool hasInput = TargetObjects.Any(o => o.Points.Count > 0 || o.Boxes.Count > 0) || !string.IsNullOrWhiteSpace(textPrompt);

        if (!hasInput)
        {
            MessageBox.Show("切り抜く対象をクリック/ドラッグするか、テキストプロンプトを入力してください。", "エラー", MessageBoxButton.OK, MessageBoxImage.Warning);
            return;
        }

        IsConfirmed = true;
        Close();
    }

    private void Mode_Checked(object sender, RoutedEventArgs e)
    {
        if (_isUpdatingMode) return;
        if (PointModeRadioButton == null || BoxModeRadioButton == null || PromptTextBox == null) return;
        _isUpdatingMode = true;
        try
        {
            ClearInputs();
            if (PointModeRadioButton.IsChecked == true) PromptTextBox.Text = "";
        }
        finally { _isUpdatingMode = false; }
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
                finally { _isUpdatingMode = false; }
            }
        }
    }

    private void RedrawMarkers()
    {
        MarkerCanvas.Children.Clear();
        _drawnBoxShapes.Clear();

        foreach (var obj in TargetObjects)
        {
            var brush = GetBrushForObject(obj.Id, true);

            foreach (var p in obj.Points)
            {
                bool isPositive = p.Item3 == 1;
                var pointBrush = isPositive ? brush : Brushes.Red;

                var ellipse = new Ellipse
                {
                    Width = 10,
                    Height = 10,
                    Fill = pointBrush,
                    Stroke = Brushes.White,
                    StrokeThickness = 2,
                    Opacity = obj == _currentObject ? 1.0 : 0.4
                };
                Canvas.SetLeft(ellipse, p.Item1 - 5);
                Canvas.SetTop(ellipse, p.Item2 - 5);
                MarkerCanvas.Children.Add(ellipse);
            }

            foreach (var b in obj.Boxes)
            {
                var rect = new Rectangle
                {
                    Stroke = brush,
                    StrokeThickness = 2,
                    Fill = new SolidColorBrush(Color.FromArgb(0x15, ObjectColors[(obj.Id - 1) % ObjectColors.Length].R, ObjectColors[(obj.Id - 1) % ObjectColors.Length].G, ObjectColors[(obj.Id - 1) % ObjectColors.Length].B)),
                    Opacity = obj == _currentObject ? 1.0 : 0.4
                };
                Canvas.SetLeft(rect, b.Item1);
                Canvas.SetTop(rect, b.Item2);
                rect.Width = b.Item3 - b.Item1;
                rect.Height = b.Item4 - b.Item2;
                MarkerCanvas.Children.Add(rect);

                if (obj == _currentObject)
                {
                    _drawnBoxShapes.Add(rect);
                }
            }
        }
    }

    private void AddObjectButton_Click(object sender, RoutedEventArgs e)
    {
        _isInitializingObjects = true;
        int nextId = TargetObjects.Count > 0 ? TargetObjects.Max(o => o.Id) + 1 : 1;
        var newObj = new TargetObject { Id = nextId };

        // ★追加：オブジェクト追加時に、オプションとして名前を入力させる可愛いインプットダイアログを表示
        string inputName = Microsoft.VisualBasic.Interaction.InputBox(
            $"新しく追加するオブジェクトの名称を入力してください。\n空欄のままでも自動で「オブジェクト {nextId}」としてナンバリング登録されます。",
            "オブジェクトの追加",
            $"オブジェクト {nextId}");

        if (!string.IsNullOrWhiteSpace(inputName))
        {
            newObj.CustomName = inputName.Trim();
        }

        TargetObjects.Add(newObj);

        ObjectComboBox.ItemsSource = null;
        ObjectComboBox.ItemsSource = TargetObjects;
        ObjectComboBox.SelectedItem = newObj;
        _currentObject = newObj;
        _isInitializingObjects = false;

        RedrawMarkers();
    }

    // ★追加：選択したオブジェクトを丸ごとリストから完全削除するメソッド（プレビュー同期ロード付き）
    private async void RemoveObjectButton_Click(object sender, RoutedEventArgs e)
    {
        if (_currentObject == null) return;

        // オブジェクトが残り1つの場合は削除できないようにガード
        if (TargetObjects.Count <= 1)
        {
            MessageBox.Show("最低1つの追跡対象オブジェクトが必要です。これ以上削除できません。", "情報", MessageBoxButton.OK, MessageBoxImage.Information);
            return;
        }

        var result = MessageBox.Show($"選択中の「{_currentObject.Name}」を完全に削除しますか？\n登録されていたポイントや箱の情報もすべてクリアされます。", "オブジェクトの削除確認", MessageBoxButton.YesNo, MessageBoxImage.Question);
        if (result != MessageBoxResult.Yes) return;

        _isInitializingObjects = true;

        var toRemove = _currentObject;
        int removeIndex = TargetObjects.IndexOf(toRemove);
        TargetObjects.Remove(toRemove);

        // 新しいアクティブオブジェクトを決定
        int newIndex = Math.Max(0, removeIndex - 1);
        var nextActive = TargetObjects[newIndex];

        ObjectComboBox.ItemsSource = null;
        ObjectComboBox.ItemsSource = TargetObjects;
        ObjectComboBox.SelectedItem = nextActive;
        _currentObject = nextActive;

        _isInitializingObjects = false;

        RedrawMarkers();

        // 削除に伴い、プレビュー上の半透明マスクも自動でリロード
        await UpdatePreviewAsync(CancellationToken.None);
    }

    private void ObjectComboBox_SelectionChanged(object sender, SelectionChangedEventArgs e)
    {
        if (_isInitializingObjects) return;
        var selected = ObjectComboBox.SelectedItem as TargetObject;
        if (selected != null)
        {
            _currentObject = selected;
            RedrawMarkers();
        }
    }

    private void DeleteObjectButton_Click(object sender, RoutedEventArgs e)
    {
        if (_currentObject == null) return;
        _currentObject.Points.Clear();
        _currentObject.Boxes.Clear();
        RedrawMarkers();
    }

    public Dictionary<string, object> GetObjectsData()
    {
        var objectsData = new Dictionary<string, object>();
        foreach (var obj in TargetObjects)
        {
            if (obj.Points.Count == 0 && obj.Boxes.Count == 0) continue;

            objectsData[obj.Id.ToString()] = new
            {
                points = obj.Points.Select(p => new int[] { p.Item1, p.Item2 }).ToArray(),
                labels = obj.Points.Select(p => p.Item3).ToArray(),
                boxes = obj.Boxes.Select(b => new int[] { b.Item1, b.Item2, b.Item3, b.Item4 }).ToArray()
            };
        }
        return objectsData;
    }

    private void ClearInputs()
    {
        foreach (var obj in TargetObjects)
        {
            obj.Points.Clear();
            obj.Boxes.Clear();
        }
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
}