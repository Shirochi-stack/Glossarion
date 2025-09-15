Set WshShell = CreateObject("WScript.Shell")
WshShell.Run "pythonw.exe translator_gui.py", 0
Set WshShell = Nothing
