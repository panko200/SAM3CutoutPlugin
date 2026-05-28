using System;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Net.Http;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;

namespace SAM3CutoutPlugin;

/// <summary>
/// HuggingFace から SAM3 ONNX モデルを自動ダウンロードするヘルパー。
/// vietanhdev/segment-anything-3-onnx-models リポジトリ（gated認証不要）から取得。
/// sam3_vit_h.zip (約3.4GB) を1ファイルでダウンロードし展開する。
/// </summary>
internal static class ModelDownloader
{
    private const string HF_REPO = "vietanhdev/segment-anything-3-onnx-models";
    private const string ZIP_FILE = "sam3_vit_h.zip";
    private const string DOWNLOAD_URL = $"https://huggingface.co/{HF_REPO}/resolve/main/{ZIP_FILE}";

    // SAM3 の3つのONNXモデルファイル
    public static readonly string[] ModelFiles = new[]
    {
        "sam3_image_encoder.onnx",
        "sam3_language_encoder.onnx",
        "sam3_decoder.onnx",
    };

    /// <summary>
    /// プラグインフォルダ内の models/ ディレクトリパスを返す
    /// </summary>
    public static string GetModelDirectory()
    {
        string pluginDir = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location) ?? "";
        return Path.Combine(pluginDir, "models");
    }

    /// <summary>
    /// 全モデルファイルが揃っているか確認
    /// </summary>
    public static bool AreModelsAvailable()
    {
        string modelDir = GetModelDirectory();
        foreach (var file in ModelFiles)
        {
            // models/ 直下、または models/sam3_vit_h/ 下にある可能性がある
            if (!File.Exists(Path.Combine(modelDir, file)) &&
                !File.Exists(Path.Combine(modelDir, "sam3_vit_h", file)))
                return false;
        }
        return true;
    }

    /// <summary>
    /// 実際のモデルファイルパスを返す（models/ 直下か models/sam3_vit_h/ 下を探す）
    /// </summary>
    public static string GetModelFilePath(string fileName)
    {
        string modelDir = GetModelDirectory();
        string direct = Path.Combine(modelDir, fileName);
        if (File.Exists(direct)) return direct;

        string subDir = Path.Combine(modelDir, "sam3_vit_h", fileName);
        if (File.Exists(subDir)) return subDir;

        // さらにサブフォルダを再帰的に検索
        foreach (var found in Directory.EnumerateFiles(modelDir, fileName, SearchOption.AllDirectories))
            return found;

        return direct; // 見つからない場合はデフォルト
    }

    /// <summary>
    /// モデルが実際に配置されているディレクトリを返す
    /// </summary>
    public static string GetActualModelDirectory()
    {
        string modelDir = GetModelDirectory();

        // 直下にあるか
        if (File.Exists(Path.Combine(modelDir, ModelFiles[0])))
            return modelDir;

        // sam3_vit_h サブフォルダ
        string subDir = Path.Combine(modelDir, "sam3_vit_h");
        if (File.Exists(Path.Combine(subDir, ModelFiles[0])))
            return subDir;

        // 再帰検索
        foreach (var found in Directory.EnumerateFiles(modelDir, ModelFiles[0], SearchOption.AllDirectories))
            return Path.GetDirectoryName(found) ?? modelDir;

        return modelDir;
    }

    /// <summary>
    /// ZIPファイルをダウンロードして展開する
    /// </summary>
    public static async Task DownloadModelsAsync(
        IProgress<(string fileName, double progress)>? progress = null,
        CancellationToken ct = default)
    {
        string modelDir = GetModelDirectory();
        Directory.CreateDirectory(modelDir);

        string zipPath = Path.Combine(modelDir, ZIP_FILE);
        string tempZipPath = zipPath + ".tmp";

        try
        {
            // ---- Step 1: ZIPをダウンロード ----
            using var httpClient = new HttpClient();
            httpClient.Timeout = TimeSpan.FromMinutes(60);

            Debug.WriteLine($"Downloading {ZIP_FILE} from {DOWNLOAD_URL}");
            progress?.Report(($"{ZIP_FILE} ダウンロード開始...", 0.0));

            using var response = await httpClient.GetAsync(DOWNLOAD_URL, HttpCompletionOption.ResponseHeadersRead, ct);
            response.EnsureSuccessStatusCode();

            long? totalBytes = response.Content.Headers.ContentLength;

            using (var contentStream = await response.Content.ReadAsStreamAsync(ct))
            using (var fileStream = new FileStream(tempZipPath, FileMode.Create, FileAccess.Write, FileShare.None, 81920))
            {
                var buffer = new byte[81920];
                long totalRead = 0;
                int bytesRead;

                while ((bytesRead = await contentStream.ReadAsync(buffer, ct)) > 0)
                {
                    ct.ThrowIfCancellationRequested();
                    await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead), ct);
                    totalRead += bytesRead;

                    if (totalBytes > 0)
                    {
                        double pct = (double)totalRead / totalBytes.Value;
                        string sizeInfo = $"{totalRead / (1024 * 1024)}MB / {totalBytes.Value / (1024 * 1024)}MB";
                        progress?.Report(($"ダウンロード中: {sizeInfo}", pct * 0.9));
                    }
                }
            }

            // ダウンロード完了
            File.Move(tempZipPath, zipPath, overwrite: true);

            // ---- Step 2: ZIPを展開 ----
            progress?.Report(("ZIPを展開中...", 0.92));
            using (var archive = ZipFile.OpenRead(zipPath))
            {
                foreach (var entry in archive.Entries)
                {
                    // ディレクトリエントリはスキップ
                    if (string.IsNullOrEmpty(entry.Name)) continue;

                    string destPath = Path.Combine(modelDir, entry.FullName);
                    string? destDir = Path.GetDirectoryName(destPath);
                    if (destDir != null) Directory.CreateDirectory(destDir);

                    entry.ExtractToFile(destPath, overwrite: true);
                }
            }

            // ---- Step 3: ZIP本体を削除（容量節約） ----
            progress?.Report(("クリーンアップ中...", 0.98));
            try { File.Delete(zipPath); } catch { /* ignore */ }

            progress?.Report(("モデルのダウンロードが完了しました！", 1.0));
        }
        catch
        {
            // 途中で失敗した場合はtempファイルを掃除
            try { if (File.Exists(tempZipPath)) File.Delete(tempZipPath); } catch { }
            throw;
        }
    }
}
