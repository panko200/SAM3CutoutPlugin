#pragma warning disable CS0618
using SharpGen.Runtime;
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.IO;
using System.Numerics; // ★ Matrix3x2 のために追加
using System.Windows.Media.Imaging;
using Vortice.Direct2D1;
using Vortice.Direct2D1.Effects; // ★ AffineTransform2D のために追加
using Vortice.Mathematics;
using YukkuriMovieMaker.Commons;
using YukkuriMovieMaker.Controls;
using YukkuriMovieMaker.Exo;
using YukkuriMovieMaker.Player.Video;
using YukkuriMovieMaker.Plugin;
using YukkuriMovieMaker.Plugin.Effects;
using YukkuriMovieMaker.Settings;

namespace SAM3CutoutPlugin;

[Obsolete("Internal use only", false)]
[VideoEffect("SAM3画像マスク", ["描画"], ["sam3", "mask", "画像マスク"])]
public class SAM3ImageMaskEffect : VideoEffectBase
{
    public override string Label => "SAM3画像マスク";

    [Display(GroupName = "設定", Name = "マスク画像", Description = "白黒のマスク画像ファイルへのパス")]
    [FileSelector(FileGroupType.ImageItem)]
    public string MaskImagePath { get; set; } = "";

    [Display(GroupName = "設定", Name = "反転", Description = "マスクの白黒を反転します")]
    [ToggleSlider]
    public bool Invert { get; set; } = false;

    public override IEnumerable<string> CreateExoVideoFilters(
        int keyFrameIndex, ExoOutputDescription exoOutputDescription) => [];

    public override IVideoEffectProcessor CreateVideoEffect(IGraphicsDevicesAndContext devices)
        => new SAM3ImageMaskEffectProcessor(devices, this);

    protected override IEnumerable<IAnimatable> GetAnimatables() => [];
}

// ─── プロセッサ ─────────────────────────────────────────────────────

internal class SAM3ImageMaskEffectProcessor : IVideoEffectProcessor, IDisposable
{
    private readonly IGraphicsDevicesAndContext _devices;
    private readonly SAM3ImageMaskEffect _effect;
    private ID2D1Image? _input;

    private string _lastImagePath = "";
    private ID2D1Bitmap? _maskBitmap;
    private ColorMatrix? _colorMatrixEffect;
    private AffineTransform2D? _transformEffect; // ★ 平行移動用エフェクトを追加
    private AlphaMask? _alphaMaskEffect;

    public SAM3ImageMaskEffectProcessor(IGraphicsDevicesAndContext devices, SAM3ImageMaskEffect effect)
    {
        _devices = devices;
        _effect = effect;
    }

    public DrawDescription Update(EffectDescription desc)
    {
        return desc.DrawDescription;
    }

    public ID2D1Image Output
    {
        get
        {
            if (string.IsNullOrWhiteSpace(_effect.MaskImagePath) || _input == null)
                return _input!;

            var dc = _devices.DeviceContext;

            // 画像のロード（パスが変わった場合のみ再読み込み）
            if (_effect.MaskImagePath != _lastImagePath)
            {
                _maskBitmap?.Dispose();
                _maskBitmap = null;
                _lastImagePath = _effect.MaskImagePath;

                if (File.Exists(_effect.MaskImagePath))
                {
                    _maskBitmap = LoadD2D1Bitmap(_effect.MaskImagePath, dc);
                }
            }

            if (_maskBitmap == null)
                return _input;

            try
            {
                if (_colorMatrixEffect == null)
                {
                    _colorMatrixEffect = new ColorMatrix(dc);
                    _colorMatrixEffect.AlphaMode = ColorMatrixAlphaMode.Straight;
                }
                if (_transformEffect == null)
                {
                    _transformEffect = new AffineTransform2D(dc);
                }
                if (_alphaMaskEffect == null)
                {
                    _alphaMaskEffect = new AlphaMask(dc);
                }

                // RGBのRedチャンネルをAlphaにマッピングし、必要に応じて反転する行列
                var matrix = _effect.Invert ? new Matrix5x4
                {
                    M11 = 1f,
                    M22 = 1f,
                    M33 = 1f,
                    M44 = 0f,
                    M14 = -1f,
                    M54 = 1f
                } : new Matrix5x4
                {
                    M11 = 1f,
                    M22 = 1f,
                    M33 = 1f,
                    M44 = 0f,
                    M14 = 1f,
                    M54 = 0f
                };
                _colorMatrixEffect.Matrix = matrix;
                _colorMatrixEffect.SetInput(0, _maskBitmap, (RawBool)true);

                // マスク画像の位置を中心に合わせるため、[-Width/2, -Height/2] だけ平行移動させる
                float w = _maskBitmap.PixelSize.Width;
                float h = _maskBitmap.PixelSize.Height;
                var translation = Matrix3x2.CreateTranslation(-w / 2f, -h / 2f);
                _transformEffect.TransformMatrix = translation;

                // 【修正】_colorMatrixEffect.Output の代わりに、_colorMatrixEffect を直接セットします
                _transformEffect.SetInputEffect(0, _colorMatrixEffect, (RawBool)true);

                _alphaMaskEffect.SetInput(0, _input, (RawBool)true);

                // 【修正】_transformEffect.Output の代わりに、_transformEffect を直接セットします
                _alphaMaskEffect.SetInputEffect(1, _transformEffect, (RawBool)true);

                return _alphaMaskEffect.Output; // 最終出力のみ、YMM4が正常に管理して解放してくれます
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"SAM3ImageMaskEffect render error: {ex.Message}");
            }

            return _input;
        }
    }

    private ID2D1Bitmap? LoadD2D1Bitmap(string path, ID2D1DeviceContext dc)
    {
        try
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

            var size = new SizeI(w, h);

            // デバイスコンテキストの現在のDPIを取得して適用（DPI倍率によるスケールズレ防止）
            dc.GetDpi(out float dpiX, out float dpiY);

            var properties = new BitmapProperties1
            {
                PixelFormat = new Vortice.DCommon.PixelFormat(Vortice.DXGI.Format.B8G8R8A8_UNorm, Vortice.DCommon.AlphaMode.Premultiplied),
                DpiX = dpiX,
                DpiY = dpiY,
                BitmapOptions = BitmapOptions.None
            };

            unsafe
            {
                fixed (byte* p = pixels)
                {
                    return dc.CreateBitmap(size, (IntPtr)p, stride, properties);
                }
            }
        }
        catch (Exception ex)
        {
            System.Diagnostics.Debug.WriteLine($"Failed to load bitmap: {ex.Message}");
            return null;
        }
    }

    public void SetInput(ID2D1Image? input) => _input = input;
    public void ClearInput() => _input = null;

    public void Dispose()
    {
        _maskBitmap?.Dispose();
        _maskBitmap = null;
        _colorMatrixEffect?.Dispose();
        _colorMatrixEffect = null;
        _transformEffect?.Dispose(); // ★ 解放
        _transformEffect = null;
        _alphaMaskEffect?.Dispose();
        _alphaMaskEffect = null;
    }
}