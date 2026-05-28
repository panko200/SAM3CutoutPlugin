using System;
using System.Diagnostics;
using System.IO;
using System.IO.Compression; // ZipFileを使用するために追加
using System.Linq;           // FirstOrDefaultを使用するために追加
using System.Net.Http;       // HttpClientを使用するために追加
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;

namespace SAM3CutoutPlugin;

/// <summary>
/// ffmpegを使って映像をフレーム分解や情報取得を行う処理クラス。
/// </summary>
internal class VideoProcessor
{
    public VideoProcessor()
    {
    }

    /// <summary>
    /// 最初のフレームを抽出してパスを返します。
    /// </summary>
    public async Task<string> ExtractFirstFrameAsync(string inputVideoPath, CancellationToken ct)
    {
        // 呼び出し時点で確実に存在しているパスを取得
        string ffmpegPath = TryFindFfmpeg() ?? throw new FileNotFoundException("ffmpeg.exe が配置されていません。");
        string tempDir = Path.Combine(Path.GetTempPath(), $"SAM3Cutout_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);
        string outputPath = Path.Combine(tempDir, "frame.png");

        string extractArgs = $"-y -i \"{inputVideoPath}\" -vframes 1 \"{outputPath}\"";
        await RunFfmpegAsync(ffmpegPath, extractArgs, ct);

        return outputPath;
    }

    #region Image helpers

    /// <summary>
    /// 画像ファイルをRGB byte配列として読み込む
    /// </summary>
    public static (byte[] rgbPixels, int width, int height) LoadImageAsRgb(string path)
    {
        using var stream = File.OpenRead(path);
        var decoder = BitmapDecoder.Create(stream, BitmapCreateOptions.PreservePixelFormat, BitmapCacheOption.OnLoad);
        var frame = decoder.Frames[0];

        // BGRA32に変換
        var bgra = new FormatConvertedBitmap(frame, System.Windows.Media.PixelFormats.Bgra32, null, 0);
        int w = bgra.PixelWidth;
        int h = bgra.PixelHeight;
        int stride = w * 4;
        var pixels = new byte[stride * h];
        bgra.CopyPixels(pixels, stride, 0);

        // BGRA → RGB
        var rgb = new byte[w * h * 3];
        for (int i = 0; i < w * h; i++)
        {
            rgb[i * 3] = pixels[i * 4 + 2];     // R
            rgb[i * 3 + 1] = pixels[i * 4 + 1]; // G
            rgb[i * 3 + 2] = pixels[i * 4];     // B
        }

        return (rgb, w, h);
    }

    /// <summary>
    /// RGB画像にマスクを適用して透過PNGとして保存
    /// </summary>
    public static void SaveMaskedImage(byte[] rgbPixels, byte[] mask, int width, int height, string outputPath)
    {
        var bgra = new byte[width * height * 4];
        for (int i = 0; i < width * height; i++)
        {
            bgra[i * 4] = rgbPixels[i * 3 + 2];     // B
            bgra[i * 4 + 1] = rgbPixels[i * 3 + 1]; // G
            bgra[i * 4 + 2] = rgbPixels[i * 3];     // R
            bgra[i * 4 + 3] = mask[i];               // A (0=透過, 255=不透明)
        }

        var bitmap = BitmapSource.Create(
            width, height, 96, 96,
            System.Windows.Media.PixelFormats.Bgra32, null,
            bgra, width * 4);

        using var fs = new FileStream(outputPath, FileMode.Create);
        var encoder = new PngBitmapEncoder();
        encoder.Frames.Add(BitmapFrame.Create(bitmap));
        encoder.Save(fs);
    }

    #endregion

    #region ffmpeg helpers

    public static async Task<string> GetOrDownloadFfmpegAsync(IProgress<(string message, double progress)>? progress, CancellationToken ct)
    {
        string? existingPath = TryFindFfmpeg();
        if (existingPath != null)
        {
            return existingPath;
        }

        progress?.Report(("ffmpeg.exe が見つかりません。自動ダウンロードを開始します...", 0.0));

        string pluginDir = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location) ?? "";
        string ffmpegDir = Path.Combine(pluginDir, "ffmpeg");
        Directory.CreateDirectory(ffmpegDir);
        string targetExePath = Path.Combine(ffmpegDir, "ffmpeg.exe");

        string zipUrl = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip";
        string tempZipPath = Path.Combine(Path.GetTempPath(), $"ffmpeg_{Guid.NewGuid():N}.zip");

        try
        {
            using var client = new HttpClient();
            client.Timeout = TimeSpan.FromMinutes(10);

            // 1. ダウンロード
            using (var response = await client.GetAsync(zipUrl, HttpCompletionOption.ResponseHeadersRead, ct))
            {
                response.EnsureSuccessStatusCode();
                long? totalBytes = response.Content.Headers.ContentLength;

                using var contentStream = await response.Content.ReadAsStreamAsync(ct);
                using var fileStream = new FileStream(tempZipPath, FileMode.Create, FileAccess.Write, FileShare.None, 81920);

                byte[] buffer = new byte[81920];
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
                        progress?.Report(($"ffmpegダウンロード中: {sizeInfo}", pct * 0.9));
                    }
                }
            }

            // 2. 解凍・展開
            progress?.Report(("ffmpeg.exe を展開中...", 0.95));

            await Task.Run(() =>
            {
                using var archive = ZipFile.OpenRead(tempZipPath);
                // ZIP内の "bin/ffmpeg.exe" を探して抽出
                var entry = archive.Entries.FirstOrDefault(e => e.FullName.EndsWith("bin/ffmpeg.exe", StringComparison.OrdinalIgnoreCase) || e.FullName.EndsWith("ffmpeg.exe", StringComparison.OrdinalIgnoreCase));
                if (entry != null)
                {
                    entry.ExtractToFile(targetExePath, overwrite: true);
                }
                else
                {
                    throw new FileNotFoundException("ZIP内に ffmpeg.exe が見つかりませんでした。");
                }
            }, ct);

            progress?.Report(("ffmpegの配置が完了しました。", 1.0));
            return targetExePath;
        }
        finally
        {
            try { if (File.Exists(tempZipPath)) File.Delete(tempZipPath); } catch { }
        }
    }

    private static string FindFfmpeg()
    {
        return TryFindFfmpeg() ?? throw new FileNotFoundException("ffmpeg.exeが見つかりません。");
    }

    private static string? TryFindFfmpeg()
    {
        string pluginDir = Path.GetDirectoryName(
            System.Reflection.Assembly.GetExecutingAssembly().Location) ?? "";

        // 1. プラグインフォルダ直下
        string localFfmpeg = Path.Combine(pluginDir, "ffmpeg.exe");
        if (File.Exists(localFfmpeg)) return localFfmpeg;

        // 2. プラグインフォルダ内の ffmpeg/ サブフォルダ
        string ffmpegSubDir = Path.Combine(pluginDir, "ffmpeg", "ffmpeg.exe");
        if (File.Exists(ffmpegSubDir)) return ffmpegSubDir;

        // 3. YMM4のプロセスからインストール先を特定してffmpegを探す
        try
        {
            var ymm4Process = System.Diagnostics.Process.GetProcessesByName("YukkuriMovieMaker");
            foreach (var proc in ymm4Process)
            {
                try
                {
                    string? ymm4ExeDir = Path.GetDirectoryName(proc.MainModule?.FileName);
                    if (!string.IsNullOrEmpty(ymm4ExeDir))
                    {
                        string candidate = Path.Combine(ymm4ExeDir, "ffmpeg.exe");
                        if (File.Exists(candidate)) return candidate;

                        string? found = SearchFileRecursive(ymm4ExeDir, "ffmpeg.exe", 2);
                        if (found != null) return found;
                    }
                }
                catch { }
            }
        }
        catch { }

        // 4. pluginフォルダの親階層
        try
        {
            string? dir = pluginDir;
            for (int i = 0; i < 4 && dir != null; i++)
            {
                dir = Path.GetDirectoryName(dir);
                if (dir != null)
                {
                    string candidate = Path.Combine(dir, "ffmpeg.exe");
                    if (File.Exists(candidate)) return candidate;

                    string? found = SearchFileRecursive(dir, "ffmpeg.exe", 1);
                    if (found != null) return found;
                }
            }
        }
        catch { }

        // 5. PATHにあるffmpeg
        string? pathFfmpeg = FindInPath("ffmpeg.exe");
        if (pathFfmpeg != null) return pathFfmpeg;

        return null;
    }

    private static string ReturnAndLog(string path)
    {
        Debug.WriteLine($"ffmpeg found: {path}");
        return path;
    }

    private static string? SearchFileRecursive(string dir, string fileName, int maxDepth)
    {
        if (maxDepth < 0) return null;
        try
        {
            string candidate = Path.Combine(dir, fileName);
            if (File.Exists(candidate)) return candidate;

            if (maxDepth > 0)
            {
                foreach (var subDir in Directory.GetDirectories(dir))
                {
                    string? found = SearchFileRecursive(subDir, fileName, maxDepth - 1);
                    if (found != null) return found;
                }
            }
        }
        catch { /* アクセス拒否等は無視 */ }
        return null;
    }

    private static string? FindInPath(string exe)
    {
        var paths = Environment.GetEnvironmentVariable("PATH")?.Split(Path.PathSeparator) ?? [];
        foreach (var dir in paths)
        {
            string full = Path.Combine(dir.Trim(), exe);
            if (File.Exists(full)) return full;
        }
        return null;
    }

    private static async Task RunFfmpegAsync(string ffmpegPath, string args, CancellationToken ct)
    {
        Debug.WriteLine($"ffmpeg: {ffmpegPath} {args}");
        using var process = new Process();
        process.StartInfo.FileName = ffmpegPath;
        process.StartInfo.Arguments = args;
        process.StartInfo.UseShellExecute = false;
        process.StartInfo.CreateNoWindow = true;
        process.StartInfo.RedirectStandardError = true;
        process.StartInfo.RedirectStandardOutput = true;

        process.Start();

        var stderrTask = process.StandardError.ReadToEndAsync(ct);
        var stdoutTask = process.StandardOutput.ReadToEndAsync(ct);

        await process.WaitForExitAsync(ct);

        if (process.ExitCode != 0)
        {
            string err = await stderrTask;

            // 0xC0000135 = DLL not found
            if (process.ExitCode == unchecked((int)0xC0000135))
            {
                throw new InvalidOperationException(
                    $"ffmpeg.exe の起動に失敗しました（DLLが不足しています）。\n" +
                    $"使用中: {ffmpegPath}\n\n" +
                    $"ffmpeg.exe 単体ではなく、「静的ビルド(static build)」を使用してください。\n" +
                    $"https://www.gyan.dev/ffmpeg/builds/ の\n" +
                    $"  'ffmpeg-release-essentials.zip' をダウンロードし、\n" +
                    $"  bin/ffmpeg.exe をプラグインのffmpegフォルダに配置してください。");
            }

            throw new InvalidOperationException($"ffmpeg failed (exit {process.ExitCode}):\n{err}");
        }
    }

    public async Task<double> GetVideoDurationAsync(string videoPath, CancellationToken ct)
    {
        string ffmpegPath = FindFfmpeg();
        using var process = new Process();
        process.StartInfo.FileName = ffmpegPath;
        process.StartInfo.Arguments = $"-i \"{videoPath}\"";
        process.StartInfo.UseShellExecute = false;
        process.StartInfo.CreateNoWindow = true;
        process.StartInfo.RedirectStandardError = true;
        process.Start();

        string stderr = await process.StandardError.ReadToEndAsync(ct);
        await process.WaitForExitAsync(ct);

        // "Duration: 00:00:05.12" のようなパターンから秒数を計算
        var match = System.Text.RegularExpressions.Regex.Match(stderr, @"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)");
        if (match.Success)
        {
            int hours = int.Parse(match.Groups[1].Value);
            int minutes = int.Parse(match.Groups[2].Value);
            double seconds = double.Parse(match.Groups[3].Value, System.Globalization.CultureInfo.InvariantCulture);
            return hours * 3600 + minutes * 60 + seconds;
        }
        return 0.0;
    }

    #endregion
}