#pragma warning disable CS0618
using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.IO;
using System.Numerics;
using SharpGen.Runtime;
using Vortice.Direct2D1;
using Vortice.Direct2D1.Effects;
using Vortice.Mathematics;
using YukkuriMovieMaker.Commons;
using YukkuriMovieMaker.Controls;
using YukkuriMovieMaker.Exo;
using YukkuriMovieMaker.Player.Video;
using YukkuriMovieMaker.Plugin.Effects;
using YukkuriMovieMaker.Plugin;
using YukkuriMovieMaker.Plugin.FileSource;
using YukkuriMovieMaker.Settings;

namespace SAM3CutoutPlugin;

[Obsolete("Internal use only", false)]
[VideoEffect("SAM3マスク", ["描画"], ["sam3", "mask", "マスク"])]
public class SAM3MaskEffect : VideoEffectBase
{
    public override string Label => "SAM3マスク";

    [Display(GroupName = "設定", Name = "マスク動画", Description = "白黒のマスク動画ファイルへのパス")]
    [FileSelector(FileGroupType.VideoItem)]
    public string MaskVideoPath { get; set; } = "";

    [Display(GroupName = "設定", Name = "オフセット（秒）", Description = "再生位置をずらす秒数")]
    [AnimationSlider("F2", "秒", -100.0, 100.0)]
    public Animation Offset { get; } = new Animation(0.0, -100.0, 100.0);

    [Display(GroupName = "設定", Name = "反転", Description = "マスクの白黒を反転します")]
    [ToggleSlider]
    public bool Invert { get; set; } = false;

    [Display(GroupName = "設定", Name = "開始秒（秒）", Description = "マスク再生を開始する元動画の位置")]
    [AnimationSlider("F2", "秒", 0.0, 10000.0)]
    public Animation Start { get; } = new Animation(0.0, 0.0, 10000.0);

    [Display(GroupName = "設定", Name = "終了秒（秒）", Description = "マスク再生を終了する元動画の位置（0のときは制限なし）")]
    [AnimationSlider("F2", "秒", 0.0, 10000.0)]
    public Animation End { get; } = new Animation(0.0, 0.0, 10000.0);

    public override IEnumerable<string> CreateExoVideoFilters(
        int keyFrameIndex, ExoOutputDescription exoOutputDescription) => [];

    public override IVideoEffectProcessor CreateVideoEffect(IGraphicsDevicesAndContext devices)
        => new SAM3MaskEffectProcessor(devices, this);

    protected override IEnumerable<IAnimatable> GetAnimatables() => [Offset, Start, End];
}

// ─── プロセッサ ─────────────────────────────────────────────────────

internal class SAM3MaskEffectProcessor : IVideoEffectProcessor, IDisposable
{
    private readonly IGraphicsDevicesAndContext _devices;
    private readonly SAM3MaskEffect _effect;
    private ID2D1Image? _input;

    private double _timelineFps = 30.0;
    private int _currentFrame = 0;
    private double _offsetSec = 0.0;
    private double _startSec = 0.0;
    private double _endSec = 0.0;

    private string _lastVideoPath = "";
    private IVideoFileSource? _videoSource;
    private ColorMatrix? _colorMatrixEffect;
    private AlphaMask? _alphaMaskEffect;

    public SAM3MaskEffectProcessor(IGraphicsDevicesAndContext devices, SAM3MaskEffect effect)
    {
        _devices = devices;
        _effect = effect;
    }

    public DrawDescription Update(EffectDescription desc)
    {
        _timelineFps = desc.FPS;
        _currentFrame = desc.ItemPosition.Frame;
        var len = desc.ItemDuration.Frame;
        var fps = desc.FPS;

        _offsetSec = _effect.Offset.GetValue(_currentFrame, len, fps);
        _startSec = _effect.Start.GetValue(_currentFrame, len, fps);
        _endSec = _effect.End.GetValue(_currentFrame, len, fps);

        return desc.DrawDescription;
    }

    public ID2D1Image Output
    {
        get
        {
            if (string.IsNullOrWhiteSpace(_effect.MaskVideoPath) || _input == null)
                return _input!;

            var dc = _devices.DeviceContext;

            double timelineTimeSec = (double)_currentFrame / _timelineFps;
            double targetTimeSec = timelineTimeSec + _offsetSec;

            // 開始・終了フレーム制限の処理（範囲外はマスクなしで描画）
            if (targetTimeSec < 0 || targetTimeSec < _startSec)
                return _input;
            if (_endSec > 0 && targetTimeSec > _endSec)
                return _input;

            // 1. ビデオリーダーの再初期化（パスが変わった場合のみ）
            if (_effect.MaskVideoPath != _lastVideoPath)
            {
                _videoSource?.Dispose();
                _videoSource = null;
                _lastVideoPath = _effect.MaskVideoPath;

                if (File.Exists(_effect.MaskVideoPath))
                {
                    try
                    {
                        _videoSource = VideoFileSourceFactory.Create(_devices, _effect.MaskVideoPath);
                    }
                    catch (Exception ex)
                    {
                        System.Diagnostics.Debug.WriteLine($"Failed to init VideoFileSource: {ex.Message}");
                    }
                }
            }

            if (_videoSource == null)
                return _input;

            try
            {
                _videoSource.Update(TimeSpan.FromSeconds(targetTimeSec));
                var maskImage = _videoSource.Output;
                if (maskImage != null)
                {
                    if (_colorMatrixEffect == null)
                    {
                        _colorMatrixEffect = new ColorMatrix(dc);
                        _colorMatrixEffect.AlphaMode = ColorMatrixAlphaMode.Straight;
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
                    _colorMatrixEffect.SetInput(0, maskImage, (RawBool)true);

                    _alphaMaskEffect.SetInput(0, _input, (RawBool)true);

                    // 【修正】_colorMatrixEffect.Output の代わりに、_colorMatrixEffect を直接セットします
                    _alphaMaskEffect.SetInputEffect(1, _colorMatrixEffect, (RawBool)true);

                    return _alphaMaskEffect.Output;
                }
            }
            catch (Exception ex)
            {
                System.Diagnostics.Debug.WriteLine($"SAM3MaskEffect render error: {ex.Message}");
            }

            return _input;
        }
    }

    public void SetInput(ID2D1Image? input) => _input = input;
    public void ClearInput() => _input = null;

    public void Dispose()
    {
        _videoSource?.Dispose();
        _videoSource = null;
        _colorMatrixEffect?.Dispose();
        _colorMatrixEffect = null;
        _alphaMaskEffect?.Dispose();
        _alphaMaskEffect = null;
    }
}