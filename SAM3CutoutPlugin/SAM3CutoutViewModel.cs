using Microsoft.Win32;
using SAM3CutoutPlugin;
using System;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media.Imaging;
using YukkuriMovieMaker.Commons;
using YukkuriMovieMaker.Plugin;
using YukkuriMovieMaker.Project;
using YukkuriMovieMaker.Project.Items;
using YukkuriMovieMaker.UndoRedo;

namespace SAM3CutoutPlugin;

internal class SAM3CutoutViewModel : INotifyPropertyChanged, ITimelineToolViewModel
{
    private Timeline? _timeline;
    private UndoRedoManager? _undoRedoManager;
    private CancellationTokenSource? _cts;

    // 判別対象とする主要な画像拡張子リスト
    private static readonly string[] ImageExtensions = { ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif" };

    // ============================================
    // プロパティ (SAM3CutoutSettingsと連携して保存)
    // ============================================

    public bool UseSelectedItem
    {
        get => SAM3CutoutSettings.Default.UseSelectedItem;
        set
        {
            if (SAM3CutoutSettings.Default.UseSelectedItem != value)
            {
                SAM3CutoutSettings.Default.UseSelectedItem = value;
                OnPropertyChanged(nameof(UseSelectedItem));
                OnPropertyChanged(nameof(UseFilePath));
            }
        }
    }
    public bool UseFilePath
    {
        get => !UseSelectedItem;
        set => UseSelectedItem = !value;
    }

    public string InputFilePath
    {
        get => SAM3CutoutSettings.Default.InputFilePath;
        set
        {
            if (SAM3CutoutSettings.Default.InputFilePath != value)
            {
                SAM3CutoutSettings.Default.InputFilePath = value;
                OnPropertyChanged(nameof(InputFilePath));
            }
        }
    }

    public bool IsImageMode
    {
        get => SAM3CutoutSettings.Default.IsImageMode;
        set
        {
            if (SAM3CutoutSettings.Default.IsImageMode != value)
            {
                SAM3CutoutSettings.Default.IsImageMode = value;
                OnPropertyChanged(nameof(IsImageMode));
                OnPropertyChanged(nameof(IsVideoMode));
            }
        }
    }
    public bool IsVideoMode
    {
        get => !IsImageMode;
        set => IsImageMode = !value;
    }

    // 0=CUDA, 1=CPU の Index と 設定の Enum を相互変換して保存
    public int DeviceIndex
    {
        get => SAM3CutoutSettings.Default.InferenceDevice == InferenceDevice.CUDA ? 0 : 1;
        set
        {
            var targetDevice = (value == 0) ? InferenceDevice.CUDA : InferenceDevice.CPU;
            if (SAM3CutoutSettings.Default.InferenceDevice != targetDevice)
            {
                SAM3CutoutSettings.Default.InferenceDevice = targetDevice;
                OnPropertyChanged(nameof(DeviceIndex));
            }
        }
    }

    public string OutputFolder
    {
        get => SAM3CutoutSettings.Default.OutputFolder;
        set
        {
            if (SAM3CutoutSettings.Default.OutputFolder != value)
            {
                SAM3CutoutSettings.Default.OutputFolder = value;
                OnPropertyChanged(nameof(OutputFolder));
            }
        }
    }

    // 動画出力形式 (設定の Enum と直接連携)
    public OutputFormat OutputFormat
    {
        get => SAM3CutoutSettings.Default.OutputFormat;
        set
        {
            if (SAM3CutoutSettings.Default.OutputFormat != value)
            {
                SAM3CutoutSettings.Default.OutputFormat = value;
                OnPropertyChanged(nameof(OutputFormat));
            }
        }
    }

    // 静止画出力形式 (設定の Enum と直接連携)
    public ImageOutputFormat ImageOutputFormat
    {
        get => SAM3CutoutSettings.Default.ImageOutputFormat;
        set
        {
            if (SAM3CutoutSettings.Default.ImageOutputFormat != value)
            {
                SAM3CutoutSettings.Default.ImageOutputFormat = value;
                OnPropertyChanged(nameof(ImageOutputFormat));
            }
        }
    }

    private string _statusMessage = "入力を設定して実行ボタンを押してください。";
    public string StatusMessage
    {
        get => _statusMessage;
        set { _statusMessage = value; OnPropertyChanged(nameof(StatusMessage)); }
    }

    private double _progressValue = 0.0;
    public double ProgressValue
    {
        get => _progressValue;
        set { _progressValue = value; OnPropertyChanged(nameof(ProgressValue)); }
    }

    private bool _isProcessing = false;
    public bool IsProcessing
    {
        get => _isProcessing;
        set
        {
            _isProcessing = value;
            OnPropertyChanged(nameof(IsProcessing));
            OnPropertyChanged(nameof(IsNotProcessing));
            (ExecuteCommand as ActionCommand)?.RaiseCanExecuteChanged();
            (CancelCommand as ActionCommand)?.RaiseCanExecuteChanged();
        }
    }
    public bool IsNotProcessing => !_isProcessing;

    // ============================================
    // コマンド
    // ============================================

    public ICommand ExecuteCommand { get; }
    public ICommand CancelCommand { get; }
    public ICommand BrowseInputCommand { get; }
    public ICommand BrowseOutputCommand { get; }

    public SAM3CutoutViewModel()
    {
        ExecuteCommand = new ActionCommand(
            _ => !IsProcessing,
            async _ => await ExecuteAsync()
        );
        CancelCommand = new ActionCommand(
            _ => IsProcessing,
            _ => _cts?.Cancel()
        );
        BrowseInputCommand = new ActionCommand(_ =>
        {
            var dlg = new OpenFileDialog
            {
                Title = "入力ファイルを選択",
                Filter = "画像/映像ファイル|*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.mp4;*.avi;*.mov;*.webm;*.mkv|すべてのファイル|*.*"
            };
            if (dlg.ShowDialog() == true)
                InputFilePath = dlg.FileName;
        });
        BrowseOutputCommand = new ActionCommand(_ =>
        {
            var dlg = new OpenFolderDialog
            {
                Title = "出力先フォルダを選択"
            };
            if (dlg.ShowDialog() == true)
                OutputFolder = dlg.FolderName;
        });
    }

    // ============================================
    // 実行処理
    // ============================================

    private async Task ExecuteAsync()
    {
        IsProcessing = true;
        ProgressValue = 0;
        _cts = new CancellationTokenSource();

        try
        {
            // トークンの確認
            var settings = SAM3CutoutSettings.Default;
            if (string.IsNullOrWhiteSpace(settings.HuggingFaceToken))
            {
                StatusMessage = "Hugging Face トークンを設定してください。";
                MessageBox.Show("Hugging Face トークンが設定されていません。\nYMM4の「ファイル」→「設定」→「プラグイン」設定からトークンを入力してください。", "エラー", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            // ---- Step 1: 入力ファイルパスの取得 ----
            string inputPath = ResolveInputPath(quiet: false);
            if (string.IsNullOrEmpty(inputPath) || !File.Exists(inputPath))
            {
                StatusMessage = "入力ファイルが見つかりません。";
                return;
            }

            // ---- Step 2: 出力先の決定 ----
            string outputDir = string.IsNullOrWhiteSpace(OutputFolder)
                ? Path.GetDirectoryName(inputPath) ?? ""
                : OutputFolder;
            Directory.CreateDirectory(outputDir);

            // ---- Step 3: Python環境およびffmpegの確認・構築 ----
            StatusMessage = "システム要件を確認中...";
            var envProgress = new Progress<(string message, double progress)>(p =>
            {
                StatusMessage = p.message;
                ProgressValue = p.progress * 0.1;
            });

            // ffmpegの自動確認・ダウンロード
            await VideoProcessor.GetOrDownloadFfmpegAsync(envProgress, _cts.Token);

            // Python環境の構築
            await PythonEnvManager.SetupEnvironmentAsync(envProgress, _cts.Token);

            ProgressValue = 0.1;

            // ---- Step 4: 最初のフレームを取得してクリックUIを表示 ----
            StatusMessage = "プレビューを準備中...";
            var firstFrame = await ExtractFirstFrameAsync(inputPath, _cts.Token);
            if (firstFrame == null)
            {
                StatusMessage = "動画/画像の読み込みに失敗しました。";
                return;
            }

            // Pythonスクリプトのパス（プラグインフォルダ直下）
            string pluginDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) ?? "";
            string scriptPath = Path.Combine(pluginDir, "processor.py");

            // 動画か静止画かでPython側に送信するフォーマット文字列の調整
            string outputFormatStr = settings.OutputFormat.ToString();
            if (this.IsImageMode)
            {
                outputFormatStr = settings.ImageOutputFormat.ToString();
                if (outputFormatStr == "AutoMaskPNG")
                {
                    outputFormatStr = "MaskPNG";
                }
            }
            else
            {
                if (outputFormatStr == "AutoMaskMP4")
                {
                    outputFormatStr = "MaskMP4";
                }
            }

            // 元の選択アイテム（クリップ）を事前に取得
            IItem? selectedItemForInsert = null;
            double startSec = 0;
            double endSec = 0;
            if (UseSelectedItem)
            {
                selectedItemForInsert = GetSelectedItem();
                if (selectedItemForInsert != null)
                {
                    double fps = _timeline?.VideoInfo.FPS ?? 30.0;
                    if (fps <= 0) fps = 30.0;
                    startSec = selectedItemForInsert.ContentOffset.TotalSeconds;
                    double durationInTimelineSec = selectedItemForInsert.Length / fps;
                    double playbackRate = selectedItemForInsert.PlaybackRate;
                    if (playbackRate == 0) playbackRate = 100.0;
                    double durationInSourceVideoSec = durationInTimelineSec * (playbackRate / 100.0);
                    endSec = startSec + durationInSourceVideoSec;
                }
            }

            StatusMessage = "SAM3モデルをロード中... (初回起動時は数秒～十数秒かかります)";
            try
            {
                await PythonEnvManager.StartServerAsync(scriptPath, inputPath, settings.HuggingFaceToken, this.DeviceIndex, startSec, endSec, _cts.Token);
            }
            catch (Exception ex)
            {
                StatusMessage = "モデルのロードに失敗しました。";
                throw new Exception($"Pythonサーバーの起動に失敗しました:\n{ex.Message}");
            }

            // ★追加：映像モードの場合、Pythonサーバーからトリミング開始位置の正しい最初のフレームを取得して上書きする
            if (!this.IsImageMode)
            {
                try
                {
                    StatusMessage = "トリミング開始位置のフレームを取得中...";
                    var cmdObjFrame = new
                    {
                        action = "get_frame",
                        frame_index = 0,
                        output_dir = outputDir
                    };
                    string jsonCmdFrame = JsonSerializer.Serialize(cmdObjFrame);
                    string responseJson = await PythonEnvManager.SendCommandAsync(jsonCmdFrame, null, _cts.Token);

                    if (!string.IsNullOrEmpty(responseJson))
                    {
                        using (var doc = JsonDocument.Parse(responseJson))
                        {
                            var root = doc.RootElement;
                            string b64Frame = root.GetProperty("frame").GetString() ?? "";
                            if (!string.IsNullOrEmpty(b64Frame))
                            {
                                firstFrame = LoadBitmapFromBase64(b64Frame);
                            }
                        }
                    }
                }
                catch (Exception ex)
                {
                    Debug.WriteLine($"Failed to load trimmed first frame from Python server: {ex.Message}");
                    // 取得に失敗した場合は、事前にExtractFirstFrameAsyncで取得した画像をそのままフォールバック
                }
            }

            bool isConfirmed = false;
            dynamic clickWindow;
            if (this.IsImageMode)
            {
                clickWindow = new SAM3ImageClickWindow(firstFrame, inputPath, outputDir, settings.HuggingFaceToken, outputFormatStr, scriptPath);
            }
            else
            {
                clickWindow = new SAM3VideoClickWindow(firstFrame, inputPath, outputDir, settings.HuggingFaceToken, outputFormatStr, scriptPath);
            }

            // WPFのUIスレッドでダイアログを表示
            await Application.Current.Dispatcher.InvokeAsync(() =>
            {
                clickWindow.ShowDialog();
                isConfirmed = clickWindow.IsConfirmed;
            });

            if (!isConfirmed)
            {
                StatusMessage = "キャンセルされました。";
                return;
            }

            ProgressValue = 0.15;
            StatusMessage = "SAM3で切り抜き処理中... (映像の長さに応じて時間がかかります)";

            object objectsData;
            if (this.IsImageMode)
            {
                var points = (List<Tuple<int, int, int>>)clickWindow.Points;
                var boxes = (List<Tuple<int, int, int, int>>)clickWindow.Boxes;
                var singleObj = new Dictionary<string, object> {
                    { "1", new {
                        points = points.Select(p => new int[] { p.Item1, p.Item2 }).ToArray(),
                        labels = points.Select(p => p.Item3).ToArray(),
                        boxes = boxes.Select(b => new int[] { b.Item1, b.Item2, b.Item3, b.Item4 }).ToArray()
                    }}
                };
                objectsData = singleObj;
            }
            else
            {
                objectsData = clickWindow.GetObjectsData();
            }

            string textPrompt = clickWindow.TextPrompt.Trim();
            string textMode = clickWindow.TextMode;

            int selectedFrameIndex = 0;
            if (!this.IsImageMode)
            {
                selectedFrameIndex = clickWindow.SelectedFrameIndex;
            }

            var cmdObj = new
            {
                action = "track",
                objects = objectsData,
                output_dir = outputDir,
                output_format = outputFormatStr,
                text_prompt = textPrompt,
                text_mode = textMode,
                frame_index = selectedFrameIndex
            };

            string jsonCmd = JsonSerializer.Serialize(cmdObj);

            var trackProgress = new Progress<double>(p =>
            {
                ProgressValue = 0.15 + (p * 0.8);
                StatusMessage = $"SAM3で切り抜き処理中... ({Math.Round(p * 100)}%)";
            });

            string pythonOutput = await PythonEnvManager.SendCommandAsync(jsonCmd, trackProgress, _cts.Token);

            ProgressValue = 0.95;

            // 出力結果の解析
            string outputPath = "";
            bool isVideo = false;
            bool isFramesDir = false;
            foreach (var line in pythonOutput.Split(new[] { '\r', '\n' }, StringSplitOptions.RemoveEmptyEntries))
            {
                if (line.StartsWith("OUTPUT_VIDEO:"))
                {
                    outputPath = line.Substring("OUTPUT_VIDEO:".Length).Trim();
                    isVideo = true;
                }
                else if (line.StartsWith("OUTPUT_IMAGE:"))
                {
                    outputPath = line.Substring("OUTPUT_IMAGE:".Length).Trim();
                    isVideo = false;
                }
                else if (line.StartsWith("OUTPUT_FRAMES_DIR:"))
                {
                    outputPath = line.Substring("OUTPUT_FRAMES_DIR:".Length).Trim();
                    isVideo = false;
                    isFramesDir = true;
                }
            }

            bool outputExists = !string.IsNullOrEmpty(outputPath) &&
                (isFramesDir ? Directory.Exists(outputPath) : File.Exists(outputPath));
            if (!outputExists)
            {
                throw new Exception("出力ファイルの生成に失敗しました。\n" + pythonOutput);
            }

            // フレームディレクトリ出力はタイムライン追加非対応（ユーザーに通知して終了）
            if (isFramesDir)
            {
                ProgressValue = 1.0;
                StatusMessage = $"完了！フレームを出力しました: {outputPath}";
                return;
            }

            // タイムラインに追加 (元の選択クリップ情報を一緒に渡す)
            bool added = await TryAddToTimelineAsync(inputPath, outputPath, isImage: !isVideo, settings.OutputFormat, settings.ImageOutputFormat, selectedItemForInsert);
            ProgressValue = 1.0;

            StatusMessage = added
                ? "完了！切り抜き映像をタイムラインに追加しました。"
                : $"完了！出力: {outputPath}";
        }
        catch (OperationCanceledException)
        {
            StatusMessage = "処理がキャンセルされました。";
        }
        catch (Exception ex)
        {
            StatusMessage = $"エラー: {ex.Message}";
            Debug.WriteLine($"SAM3 Cutout Error: {ex}");
        }
        finally
        {
            await PythonEnvManager.StopServerAsync();
            IsProcessing = false;
            _cts?.Dispose();
            _cts = null;
        }
    }

    private async Task<BitmapSource?> ExtractFirstFrameAsync(string inputPath, CancellationToken ct)
    {
        // 匿名関数に async を追加します
        return await Task.Run(async () =>
        {
            try
            {
                var vp = new VideoProcessor();
                // メソッドを非同期で呼び出し、ct を引数に渡します
                string framePath = await vp.ExtractFirstFrameAsync(inputPath, ct);

                var image = new BitmapImage();
                image.BeginInit();
                image.CacheOption = BitmapCacheOption.OnLoad;
                image.UriSource = new Uri(framePath);
                image.EndInit();
                image.Freeze(); // 別スレッドで作成したBitmapをUIスレッドで使えるように固定
                return image;
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Failed to extract frame: {ex.Message}");
                return null;
            }
        }, ct);
    }

    // ★追加：WebPファイル用のアニメーション検知ヘルパー（バイナリ走査）
    private static bool IsAnimatedWebp(string path)
    {
        try
        {
            using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
            {
                if (fs.Length < 100) return false;
                byte[] buffer = new byte[1024];
                int bytesRead = fs.Read(buffer, 0, buffer.Length);
                string text = System.Text.Encoding.ASCII.GetString(buffer, 0, bytesRead);
                return text.Contains("ANIM") || text.Contains("ANMF");
            }
        }
        catch
        {
            return false;
        }
    }

    // ★追加：画像ファイルのアニメーション構造検知用ヘルパー
    private static bool IsAnimatedImage(string path)
    {
        try
        {
            string ext = Path.GetExtension(path).ToLower();
            if (ext == ".webp")
            {
                return IsAnimatedWebp(path);
            }
            if (ext == ".gif")
            {
                // GIFはWPFの標準BitmapDecoderを用いてフレーム数を取得することでアニメーション判定可能
                using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read))
                {
                    var decoder = BitmapDecoder.Create(fs, BitmapCreateOptions.DelayCreation, BitmapCacheOption.None);
                    return decoder.Frames.Count > 1;
                }
            }
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Failed to analyze animated image: {ex.Message}");
        }
        return false; // PNG, JPG, BMP などの標準画像は静止画として判定
    }

    // ============================================
    // 入力パスの解決
    // ============================================

    private string ResolveInputPath(bool quiet = false)
    {
        string path = "";
        if (!UseSelectedItem)
        {
            path = InputFilePath;
        }
        else
        {
            if (_timeline == null) return "";

            try
            {
                var selectedProp = _timeline.GetType().GetProperty("SelectedItem",
                    BindingFlags.Public | BindingFlags.Instance);
                if (selectedProp == null) return "";

                var selectedItem = selectedProp.GetValue(_timeline);
                if (selectedItem == null)
                {
                    if (!quiet) StatusMessage = "タイムラインでアイテムが選択されていません。";
                    return "";
                }

                var filePathProp = selectedItem.GetType().GetProperty("FilePath",
                    BindingFlags.Public | BindingFlags.Instance);
                if (filePathProp != null)
                {
                    path = filePathProp.GetValue(selectedItem) as string ?? "";
                }
                else
                {
                    if (!quiet) StatusMessage = "選択中のアイテムにファイルパスがありません。";
                    return "";
                }
            }
            catch (Exception ex)
            {
                Debug.WriteLine($"Selected item retrieval failed: {ex.Message}");
                if (!quiet) StatusMessage = "選択中アイテムの取得に失敗しました。ファイル指定モードをお使いください。";
                return "";
            }
        }

        // ★ 拡張子および画像のアニメーション構造を検知して動作モードを自動設定
        if (!string.IsNullOrEmpty(path) && File.Exists(path))
        {
            string ext = Path.GetExtension(path).ToLower();
            if (ImageExtensions.Contains(ext))
            {
                // アニメーション画像（GIFやWebP等）は動画モード、1Fのみなら静止画モードへ自動割り当て！
                if (IsAnimatedImage(path))
                {
                    IsVideoMode = true;
                }
                else
                {
                    IsImageMode = true;
                }
            }
            else
            {
                IsVideoMode = true;
            }
        }

        return path;
    }

    private IItem? GetSelectedItem()
    {
        if (_timeline == null) return null;
        try
        {
            var selectedProp = _timeline.GetType().GetProperty("SelectedItem",
                BindingFlags.Public | BindingFlags.Instance);
            if (selectedProp == null) return null;
            return selectedProp.GetValue(_timeline) as IItem;
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"Failed to get selected item: {ex.Message}");
            return null;
        }
    }

    // ============================================
    // タイムラインへの追加
    // ============================================

    private async Task<bool> TryAddToTimelineAsync(string inputPath, string outputPath, bool isImage, OutputFormat videoFormat, ImageOutputFormat imageFormat, IItem? selectedItem)
    {
        if (_timeline == null || _undoRedoManager == null) return false;

        try
        {
            IItem item;

            // ★ 出力形式が「自動透過（マスク自動適用）」であるかどうかを判定
            bool isAutoMask = (!isImage && videoFormat == OutputFormat.AutoMaskMP4) || (isImage && imageFormat == ImageOutputFormat.AutoMaskPNG);

            // 自動エフェクト適用モードの時は、タイムラインに挿入する本体を「元の形式のファイル（inputPath）」にします
            string fileToInsert = outputPath;
            if (isImage)
            {
                if (imageFormat == ImageOutputFormat.AutoMaskPNG)
                    fileToInsert = inputPath;
            }
            else
            {
                if (videoFormat == OutputFormat.AutoMaskMP4)
                    fileToInsert = inputPath;
            }

            if (isImage)
            {
                var imageItem = new ImageItem();
                imageItem.FilePath = fileToInsert;

                // ★ AutoMask の時のみ元の分割クリップからタイムライン上の長さを100%引き継ぐ
                if (isAutoMask && selectedItem != null)
                {
                    imageItem.Length = selectedItem.Length;
                }
                else
                {
                    imageItem.Length = 150; // デフォルト5秒
                }
                item = imageItem;
            }
            else
            {
                var videoItem = new VideoItem();
                videoItem.FilePath = fileToInsert;

                double timelineFps = _timeline.VideoInfo.FPS;
                if (timelineFps <= 0) timelineFps = 30.0;

                // ★ AutoMaskの時のみ、「長さ」「再生オフセット」「再生速度」を完全コピーします
                if (isAutoMask && selectedItem != null)
                {
                    videoItem.Length = selectedItem.Length;

                    // リフレクションを用いて各プロパティを安全にクローン設定
                    var offsetProp = selectedItem.GetType().GetProperty("ContentOffset");
                    if (offsetProp != null && offsetProp.CanWrite)
                    {
                        var offsetVal = offsetProp.GetValue(selectedItem);
                        offsetProp.SetValue(videoItem, offsetVal);
                    }

                    var rateProp = selectedItem.GetType().GetProperty("PlaybackRate");
                    if (rateProp != null && rateProp.CanWrite)
                    {
                        var rateVal = rateProp.GetValue(selectedItem);
                        rateProp.SetValue(videoItem, rateVal);
                    }
                }
                else
                {
                    // ★ それ以外の形式（グリーンバック等）の時は、生成された短い動画のデュレーションを元に長さを設定
                    // 再生オフセット（ContentOffset）は0のまま初期状態で追加
                    var vp = new VideoProcessor();
                    double durationSec = await vp.GetVideoDurationAsync(fileToInsert, CancellationToken.None);

                    if (durationSec > 0)
                    {
                        videoItem.Length = (int)Math.Round(durationSec * timelineFps);
                    }
                    else
                    {
                        videoItem.Length = 300; // 取得失敗時のフォールバック
                    }
                }

                item = videoItem;
            }

#pragma warning disable CS0618
            // [動画用自動マスク]
            if (!isImage && videoFormat == OutputFormat.AutoMaskMP4 && item is VisualItem viVideo)
            {
                var maskEffect = new SAM3MaskEffect
                {
                    MaskVideoPath = outputPath // 生成された白黒マスク動画
                };
                viVideo.VideoEffects = viVideo.VideoEffects.Add(maskEffect);
            }

            // [静止画用自動マスク]
            if (isImage && imageFormat == ImageOutputFormat.AutoMaskPNG && item is VisualItem viImage)
            {
                var maskEffect = new SAM3ImageMaskEffect
                {
                    MaskImagePath = outputPath // 生成された白黒マスク画像
                };
                viImage.VideoEffects = viImage.VideoEffects.Add(maskEffect);
            }
#pragma warning restore CS0618

            // ★ 全モードにおいて、アイテムの開始位置（Frame）と配置レイヤー（Layer + 1）は、
            // 元の分割されたクリップの真上になるようにピッタリ合わせます！
            int targetFrame = _timeline.CurrentFrame;
            int targetLayer = 0;

            if (selectedItem != null)
            {
                targetFrame = selectedItem.Frame;
                targetLayer = selectedItem.Layer + 1;
            }

            var tryAddMethod = _timeline.GetType().GetMethod("TryAddItems",
                BindingFlags.Public | BindingFlags.Instance);

            if (tryAddMethod != null)
            {
                var items = new IItem[] { item };
                var parameters = tryAddMethod.GetParameters();

                if (parameters.Length == 4)
                {
                    tryAddMethod.Invoke(_timeline, new object[] { items, targetFrame, targetLayer, true });
                }
                else if (parameters.Length == 3)
                {
                    tryAddMethod.Invoke(_timeline, new object[] { items, targetFrame, targetLayer });
                }

                _undoRedoManager.Record();
                return true;
            }
        }
        catch (Exception ex)
        {
            Debug.WriteLine($"タイムライン追加失敗: {ex.Message}");
        }

        return false;
    }

    // Base64から直接非ロックでBitmapImageをビルドするヘルパー
    private static BitmapImage LoadBitmapFromBase64(string b64)
    {
        byte[] bytes = Convert.FromBase64String(b64);
        using var ms = new MemoryStream(bytes);
        var bitmap = new BitmapImage();
        bitmap.BeginInit();
        bitmap.CacheOption = BitmapCacheOption.OnLoad;
        bitmap.StreamSource = ms;
        bitmap.EndInit();
        bitmap.Freeze();
        return bitmap;
    }

    // ============================================
    // ITimelineToolViewModel
    // ============================================

    public void SetTimelineToolInfo(TimelineToolInfo info)
    {
        _timeline = info.Timeline;
        _undoRedoManager = info.UndoRedoManager;

        // ★ 初期読み込み時に、現在選択されているアイテムのモード（画像・映像）を自動検知（静かに実行）
        try
        {
            ResolveInputPath(quiet: true);
        }
        catch { }
    }

    // ============================================
    // INotifyPropertyChanged
    // ============================================

    public event PropertyChangedEventHandler? PropertyChanged;
    protected void OnPropertyChanged(string name)
        => PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(name));
}