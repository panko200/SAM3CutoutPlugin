using System;
using YukkuriMovieMaker.Plugin;

namespace SAM3CutoutPlugin;

public class SAM3CutoutPlugin : IToolPlugin
{
    public string Name => "SAM3 被写体切り抜き";
    public Type ViewModelType => typeof(SAM3CutoutViewModel);
    public Type ViewType => typeof(SAM3CutoutView);
}
