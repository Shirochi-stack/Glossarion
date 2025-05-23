' launch_translator.vbs
Option Explicit
Dim shell, fso, scriptDir

Set shell     = CreateObject("WScript.Shell")
Set fso       = CreateObject("Scripting.FileSystemObject")
scriptDir     = fso.GetParentFolderName(WScript.ScriptFullName)

' make sure we run from the script's folder
shell.CurrentDirectory = scriptDir

' 0 = hidden window, False = don't wait for exit
shell.Run "pythonw.exe ""src/translator_gui.py""", 0, False
