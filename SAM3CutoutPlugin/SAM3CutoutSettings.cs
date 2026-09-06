using System.Runtime.Serialization;
using YukkuriMovieMaker.Plugin;

namespace SAM3CutoutPlugin;

/// <summary>
/// 推論デバイス設定
/// </summary>
public enum InferenceDevice
{
    /// <summary>CPU（全環境で動作）</summary>
    CPU,
    /// <summary>CUDA GPU（NVIDIA製GPU搭載のPCで推奨）</summary>
    CUDA,
}

/// <summary>
/// 動画の処理方式（メモリ使用量とトレードオフ）
/// </summary>
public enum ProcessingMode
{
    /// <summary>自動：短い動画は従来方式（従来と同一結果）、長い動画はチャンク方式（省メモリ）に自動切替</summary>
    Auto = 0,
    /// <summary>常にチャンク方式（省メモリ・長尺動画向け）</summary>
    Chunk = 1,
    /// <summary>常に従来方式（全フレームをRAMに保持し一括処理。短い動画向け）</summary>
    Legacy = 2,
}

/// <summary>
/// 動画出力形式設定
/// </summary>
public enum OutputFormat
{
    /// <summary>クロマキー用のグリーンバック MP4 形式</summary>
    GreenScreenMP4,
    /// <summary>白黒のマスク動画（YMM4クリッピング用）</summary>
    MaskMP4,
    /// <summary>白黒マスク動画（1レイヤー自動切り抜き）</summary>
    AutoMaskMP4,
}

/// <summary>
/// 静止画出力形式設定 (追加)
/// </summary>
public enum ImageOutputFormat
{
    /// <summary>透過PNG（アルファチャンネル対応形式）</summary>
    TransparentPNG,
    /// <summary>白黒のマスク画像（PNG）</summary>
    MaskPNG,
    /// <summary>白黒マスク画像で自動切り抜き適用</summary>
    AutoMaskPNG,
}

[DataContract]
internal class SAM3CutoutSettings : SettingsBase<SAM3CutoutSettings>
{
    public override SettingsCategory Category => SettingsCategory.None;
    public override string Name => "SAM3 被写体切り抜き設定";
    public override bool HasSettingView => true;
    public override object? SettingView => new SAM3CutoutSettingsView();

    // モデルフォルダパス（空欄ならプラグインフォルダ内の models/ を使用）
    private string modelFolderPath = "";
    [DataMember]
    public string ModelFolderPath { get => modelFolderPath; set => Set(ref modelFolderPath, value); }

    // ffmpegのパス（空欄ならYMM4同梱 or PATH上のものを使用）
    private string ffmpegPath = "";
    [DataMember]
    public string FfmpegPath { get => ffmpegPath; set => Set(ref ffmpegPath, value); }

    // 推論デバイス
    private InferenceDevice inferenceDevice = InferenceDevice.CPU;
    [DataMember]
    public InferenceDevice InferenceDevice { get => inferenceDevice; set => Set(ref inferenceDevice, value); }

    // 動画処理方式（自動/チャンク/従来）
    private ProcessingMode processingMode = ProcessingMode.Auto;
    [DataMember]
    public ProcessingMode ProcessingMode { get => processingMode; set => Set(ref processingMode, value); }

    // Hugging Face トークン
    private string huggingFaceToken = "";
    [DataMember]
    public string HuggingFaceToken { get => huggingFaceToken; set => Set(ref huggingFaceToken, value); }

    // 出力形式
    private OutputFormat outputFormat = OutputFormat.GreenScreenMP4;
    [DataMember]
    public OutputFormat OutputFormat { get => outputFormat; set => Set(ref outputFormat, value); }

    // === 追加：静止画の出力形式 ===
    private ImageOutputFormat imageOutputFormat = ImageOutputFormat.TransparentPNG;
    [DataMember]
    public ImageOutputFormat ImageOutputFormat { get => imageOutputFormat; set => Set(ref imageOutputFormat, value); }


    // === 追加：ユーザー設定の保存用プロパティ ===

    private bool useSelectedItem = true;
    [DataMember]
    public bool UseSelectedItem { get => useSelectedItem; set => Set(ref useSelectedItem, value); }

    private string inputFilePath = "";
    [DataMember]
    public string InputFilePath { get => inputFilePath; set => Set(ref inputFilePath, value); }

    private bool isImageMode = true;
    [DataMember]
    public bool IsImageMode { get => isImageMode; set => Set(ref isImageMode, value); }

    private string outputFolder = "";
    [DataMember]
    public string OutputFolder { get => outputFolder; set => Set(ref outputFolder, value); }


    public override void Initialize() { }
}