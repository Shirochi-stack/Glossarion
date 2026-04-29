[Setup]
AppName=Glossarion
AppVersion=8.6.4
AppPublisher=Shirochi
AppSupportURL=https://github.com/Shirochi-stack/Glossarion

; Default installation folder
DefaultDirName={localappdata}\Glossarion
PrivilegesRequired=lowest
DefaultGroupName=Glossarion
DisableProgramGroupPage=yes
UsePreviousSetupType=no

; Output settings
OutputDir=Output
OutputBaseFilename=Glossarion_Setup_v8.6.4
Compression=lzma2/ultra64
SolidCompression=yes

; Icon settings
SetupIconFile=Halgakos.ico
UninstallDisplayIcon={app}\Glossarion.exe

[Types]
Name: "custom"; Description: "Select which version of Glossarion to install:"; Flags: iscustom

[Components]
; The 'exclusive' flag means they can only select ONE of these options at a time via radio buttons.
Name: "full"; Description: "Glossarion - Optimal for Novels (Full-featured, no Manga translation)"; Types: custom; Flags: exclusive
Name: "lite"; Description: "Glossarion Lite - Excludes the EPUB Reader"; Types: custom; Flags: exclusive
Name: "turbolite"; Description: "Glossarion TurboLite - Excludes the EPUB Reader, Vertex AI SDK, and PDF Translation"; Types: custom; Flags: exclusive
Name: "nocuda"; Description: "Glossarion NoCuda - Optimal for Manga (Full-featured with experimental features)"; Types: custom; Flags: exclusive

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Notice the 'DestName' trick! No matter which one they pick, it saves to their computer as 'Glossarion.exe'.
; This makes our shortcuts and uninstaller work perfectly regardless of their choice!
Source: "dist\Glossarion v8.6.4.exe"; DestDir: "{app}"; DestName: "Glossarion.exe"; Flags: ignoreversion; Components: full
Source: "dist\N_Glossarion_NoCuda v8.6.4.exe"; DestDir: "{app}"; DestName: "Glossarion.exe"; Flags: ignoreversion; Components: nocuda
Source: "dist\L_Glossarion_Lite v8.6.4.exe"; DestDir: "{app}"; DestName: "Glossarion.exe"; Flags: ignoreversion; Components: lite
Source: "dist\L_Glossarion_TurboLite v8.6.4.exe"; DestDir: "{app}"; DestName: "Glossarion.exe"; Flags: ignoreversion; Components: turbolite

[Icons]
Name: "{autoprograms}\Glossarion"; Filename: "{app}\Glossarion.exe"
Name: "{autodesktop}\Glossarion"; Filename: "{app}\Glossarion.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Glossarion.exe"; Description: "{cm:LaunchProgram,Glossarion}"; Flags: nowait postinstall skipifsilent

[Code]
procedure CurStepChanged(CurStep: TSetupStep);
var
  Lines: TArrayOfString;
  i: Integer;
begin
  if CurStep = ssPostInstall then
  begin
    { If config.json exists, gracefully turn off the updater flag }
    if LoadStringsFromFile(ExpandConstant('{app}\config.json'), Lines) then
    begin
      for i := 0 to GetArrayLength(Lines) - 1 do
      begin
        StringChangeEx(Lines[i], '"auto_update_check": true', '"auto_update_check": false', True);
      end;
      SaveStringsToFile(ExpandConstant('{app}\config.json'), Lines, False);
    end
    else
    begin
      { If it doesn't exist, create a tiny config file to turn off the updater }
      { Glossarion's robust config loader will automatically fill in the rest of the defaults! }
      SetArrayLength(Lines, 3);
      Lines[0] := '{';
      Lines[1] := '  "auto_update_check": false';
      Lines[2] := '}';
      SaveStringsToFile(ExpandConstant('{app}\config.json'), Lines, False);
    end;
  end;
end;
