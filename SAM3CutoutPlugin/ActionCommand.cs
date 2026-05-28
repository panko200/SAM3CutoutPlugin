using System;
using System.Windows.Input;

namespace SAM3CutoutPlugin;

/// <summary>
/// シンプルなICommand実装ヘルパー
/// </summary>
internal class ActionCommand : ICommand
{
    private readonly Func<object?, bool>? _canExecute;
    private readonly Action<object?> _execute;

    public ActionCommand(Func<object?, bool>? canExecute, Action<object?> execute)
    {
        _canExecute = canExecute;
        _execute = execute;
    }

    public ActionCommand(Action<object?> execute)
        : this(null, execute)
    {
    }

    public event EventHandler? CanExecuteChanged;

    public bool CanExecute(object? parameter) => _canExecute?.Invoke(parameter) ?? true;
    public void Execute(object? parameter) => _execute(parameter);

    public void RaiseCanExecuteChanged()
    {
        CanExecuteChanged?.Invoke(this, EventArgs.Empty);
    }
}
