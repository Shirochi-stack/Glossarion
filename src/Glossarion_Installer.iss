[Setup]
AppName=Glossarion
AppVersion=8.6.4
AppPublisher=Shirochi
AppSupportURL=https://github.com/Shirochi-stack/Glossarion

; Default installation folder
DefaultDirName={autopf}\Glossarion
DefaultGroupName=Glossarion
DisableProgramGroupPage=yes

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
Name: "turbolite"; Description: "Glossarion TurboLite - Excludes the EPUB Reader, Vertex AI SDK, and PDF generation"; Types: custom; Flags: exclusive
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
