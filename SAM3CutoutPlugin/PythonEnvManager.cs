using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;

namespace SAM3CutoutPlugin;

/// <summary>
/// Pythonの自動構築および実行を管理するクラス
/// </summary>
internal static class PythonEnvManager
{
    private static Process? _serverProcess;
    private static StreamWriter? _serverInput;
    private static StreamReader? _serverOutput;

    private static readonly string PluginDir = Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location) ?? "";
    public static readonly string PythonDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
        "YMM4_SAM3CutoutPlugin", "python");
    private static readonly string UvExePath = Path.Combine(PythonDir, "uv.exe");
    private static readonly string VenvDir = Path.Combine(PythonDir, ".venv");

    public static int TotalFrames { get; private set; } = 0;

    /// <summary>
    /// Python環境（uvと各種ライブラリ）を自動構築します。
    /// </summary>
    public static async Task SetupEnvironmentAsync(IProgress<(string message, double progress)> progress, CancellationToken ct)
    {
        Directory.CreateDirectory(PythonDir);

        if (!File.Exists(UvExePath))
        {
            progress.Report(("パッケージマネージャー (uv) をダウンロード中...", 0.1));
            await DownloadUvAsync(ct);
        }

        if (!Directory.Exists(VenvDir))
        {
            progress.Report(("Python仮想環境を作成中...", 0.1));
            await RunUvCommandAsync("venv --python 3.12", ct);
        }

        progress.Report(("必要なPythonライブラリをインストール中... (初回は数分かかります)", 0.5));

        try
        {
            await RunUvCommandAsync("pip uninstall -y torch-directml", ct);
        }
        catch { }

        try
        {
            await RunUvCommandAsync("pip install --upgrade torch==2.8.0 torchvision --index-url https://download.pytorch.org/whl/cu128", ct);
        }
        catch
        {
            try
            {
                await RunUvCommandAsync("pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124", ct);
            }
            catch
            {
                await RunUvCommandAsync("pip install --upgrade torch torchvision", ct);
            }
        }

        try
        {
            // git+https:// ではなく、GitHubが提供するZIPダウンロードのURLを直接pipに渡します
            await RunUvCommandAsync("pip install https://github.com/huggingface/transformers/archive/69f003696be95cc606ad59f518e38d728514930d.zip huggingface_hub opencv-python pillow numpy accelerate", ct);
        }
        catch
        {
            await RunUvCommandAsync("pip install transformers huggingface_hub opencv-python pillow numpy accelerate", ct);
        }

        progress.Report(("Python環境の構築が完了しました。", 1.0));
    }

    private static async Task DownloadUvAsync(CancellationToken ct)
    {
        using var client = new HttpClient();
        string url = "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip";
        var zipBytes = await client.GetByteArrayAsync(url, ct);

        string tempZip = Path.Combine(PythonDir, "uv.zip");
        await File.WriteAllBytesAsync(tempZip, zipBytes, ct);

        string extractTempDir = Path.Combine(PythonDir, "uv_temp");
        if (Directory.Exists(extractTempDir)) Directory.Delete(extractTempDir, true);

        ZipFile.ExtractToDirectory(tempZip, extractTempDir, overwriteFiles: true);

        string? uvFound = null;
        foreach (var file in Directory.GetFiles(extractTempDir, "uv.exe", SearchOption.AllDirectories))
        {
            uvFound = file;
            break;
        }

        if (uvFound != null)
        {
            File.Copy(uvFound, UvExePath, true);
        }
        else
        {
            throw new FileNotFoundException("uv.exeがZIP内に見つかりませんでした。");
        }

        File.Delete(tempZip);
        Directory.Delete(extractTempDir, true);
    }

    private static async Task<string> RunUvCommandAsync(string arguments, CancellationToken ct)
    {
        using var process = new Process();
        process.StartInfo.FileName = UvExePath;
        process.StartInfo.Arguments = arguments;
        process.StartInfo.WorkingDirectory = PythonDir;
        process.StartInfo.UseShellExecute = false;
        process.StartInfo.CreateNoWindow = true;
        process.StartInfo.RedirectStandardOutput = true;
        process.StartInfo.RedirectStandardError = true;

        process.Start();

        var stdoutTask = process.StandardOutput.ReadToEndAsync(ct);
        var stderrTask = process.StandardError.ReadToEndAsync(ct);

        await process.WaitForExitAsync(ct);

        string stdout = await stdoutTask;
        string stderr = await stderrTask;

        if (process.ExitCode != 0)
        {
            throw new Exception($"Pythonコマンドの実行が失敗しました (uv {arguments}):\n{stderr}");
        }

        return stdout;
    }

    public static async Task StartServerAsync(string scriptPath, string videoPath, string token, int deviceIndex, double startSec, double endSec, CancellationToken ct)
    {
        if (_serverProcess != null && !_serverProcess.HasExited)
            return;

        _serverProcess = new Process();
        _serverProcess.StartInfo.FileName = UvExePath;
        _serverProcess.StartInfo.Arguments = $"run \"{scriptPath}\" server \"{videoPath}\" \"{token}\" \"{deviceIndex}\" \"{startSec}\" \"{endSec}\"";
        _serverProcess.StartInfo.WorkingDirectory = PythonDir;
        _serverProcess.StartInfo.UseShellExecute = false;
        _serverProcess.StartInfo.CreateNoWindow = true;
        _serverProcess.StartInfo.RedirectStandardInput = true;
        _serverProcess.StartInfo.RedirectStandardOutput = true;
        _serverProcess.StartInfo.RedirectStandardError = true;
        _serverProcess.StartInfo.StandardOutputEncoding = System.Text.Encoding.UTF8;

        _serverProcess.ErrorDataReceived += (sender, e) =>
        {
            if (e.Data != null)
                System.Diagnostics.Debug.WriteLine($"[SAM3 Python stderr] {e.Data}");
        };

        _serverProcess.Start();
        _serverProcess.BeginErrorReadLine();

        _serverInput = _serverProcess.StandardInput;
        _serverOutput = _serverProcess.StandardOutput;

        TotalFrames = 0;
        while (true)
        {
            if (ct.IsCancellationRequested)
                throw new TaskCanceledException();

            string? line = await _serverOutput.ReadLineAsync();
            if (line == null)
            {
                throw new Exception("Pythonサーバーが起動前に終了しました。(コンソール出力またはデバッグログを確認してください)");
            }

            if (line.StartsWith("TOTAL_FRAMES:"))
            {
                if (int.TryParse(line.Substring("TOTAL_FRAMES:".Length).Trim(), out int tf))
                {
                    TotalFrames = tf;
                }
            }

            if (line == "SERVER_READY")
                break;
            if (line.StartsWith("ERROR:"))
                throw new Exception(line.Substring(6));
        }
    }

    // ★修正：リクエストID紐付け通信メカニズムを搭載
    public static async Task<string> SendCommandAsync(string jsonCommand, IProgress<double>? progress, CancellationToken ct)
    {
        if (_serverProcess == null || _serverProcess.HasExited || _serverInput == null || _serverOutput == null)
            throw new Exception("Pythonサーバーが起動していません。");

        // 1. 今回のリクエスト専用の一意のIDを発行
        string requestId = Guid.NewGuid().ToString("N");

        // 2. 送信データJSONに request_id を動的に注入
        string jsonWithId;
        using (var doc = System.Text.Json.JsonDocument.Parse(jsonCommand))
        {
            var root = doc.RootElement;
            var dict = root.EnumerateObject().ToDictionary(x => x.Name, x => x.Value);

            var cmdObj = new System.Dynamic.ExpandoObject() as IDictionary<string, object>;
            foreach (var kvp in dict)
            {
                cmdObj[kvp.Key] = GetValueFromJsonElement(kvp.Value);
            }
            cmdObj["request_id"] = requestId; // IDを注入
            jsonWithId = System.Text.Json.JsonSerializer.Serialize(cmdObj);
        }

        await _serverInput.WriteLineAsync(jsonWithId);
        await _serverInput.FlushAsync();

        // 3. レスポンス待機・古いパケットの読み飛ばしループ
        while (true)
        {
            if (ct.IsCancellationRequested)
                throw new TaskCanceledException();

            string? line = await _serverOutput.ReadLineAsync();
            if (line == null)
                throw new Exception("Pythonサーバーが予期せず終了しました。");

            // 進捗報告はそのまま通す
            if (line.StartsWith("PROGRESS:"))
            {
                if (progress != null)
                {
                    string[] parts = line.Substring("PROGRESS:".Length).Split('/');
                    if (parts.Length == 2 && double.TryParse(parts[0], out double current) && double.TryParse(parts[1], out double total) && total > 0)
                    {
                        progress.Report(current / total);
                    }
                }
                continue;
            }

            // レスポンスパケットを解析。形式: PREFIX:request_id:data
            int firstColon = line.IndexOf(':');
            if (firstColon == -1) continue;

            string prefix = line.Substring(0, firstColon);
            string remainder = line.Substring(firstColon + 1);

            int secondColon = remainder.IndexOf(':');
            if (secondColon == -1) continue;

            string respId = remainder.Substring(0, secondColon);
            string data = remainder.Substring(secondColon + 1);

            // ★ 自分が送信した request_id と一致しない古いデータは、ログを吐くことなく安全に無視（破棄）して読み飛ばす！
            if (respId != requestId)
            {
                System.Diagnostics.Debug.WriteLine($"[SAM3 C#] 古いパケット({prefix})を自動破棄しました。ID: {respId}");
                continue;
            }

            // IDが一致した正しい返答のみを返す
            if (prefix == "GET_FRAME_RESPONSE" || prefix == "PREVIEW_RESPONSE")
                return data.Trim();
            if (prefix == "OUTPUT_VIDEO" || prefix == "OUTPUT_IMAGE" || prefix == "OUTPUT_FRAMES_DIR")
                return $"{prefix}:{data.Trim()}";
            if (prefix == "ERROR")
                throw new Exception(data.Trim());
        }
    }

    private static object GetValueFromJsonElement(System.Text.Json.JsonElement element)
    {
        switch (element.ValueKind)
        {
            case System.Text.Json.JsonValueKind.String: return element.GetString() ?? "";
            case System.Text.Json.JsonValueKind.Number:
                if (element.TryGetInt32(out int i)) return i;
                return element.GetDouble();
            case System.Text.Json.JsonValueKind.True: return true;
            case System.Text.Json.JsonValueKind.False: return false;
            case System.Text.Json.JsonValueKind.Null: return null!;
            case System.Text.Json.JsonValueKind.Array:
                return element.EnumerateArray().Select(GetValueFromJsonElement).ToArray();
            case System.Text.Json.JsonValueKind.Object:
                return element.EnumerateObject().ToDictionary(x => x.Name, x => GetValueFromJsonElement(x.Value));
            default: return element.ToString();
        }
    }

    public static void StopServer()
    {
        if (_serverProcess != null && !_serverProcess.HasExited)
        {
            try
            {
                _serverInput?.WriteLine("{\"action\":\"exit\"}");
                _serverInput?.Flush();
                _serverProcess.WaitForExit(1000);
                if (!_serverProcess.HasExited)
                    _serverProcess.Kill();
            }
            catch { }
        }
        _serverProcess?.Dispose();
        _serverProcess = null;
        _serverInput = null;
        _serverOutput = null;
    }
}