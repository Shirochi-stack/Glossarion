# Glossarion MSIX Packaging

This creates an MSIX package from the PyInstaller output produced by
`src/translator.spec`.

## Store package

Use this when submitting an MSIX package to Microsoft Store. It intentionally
does not sign the package; the Store re-signs accepted MSIX submissions.

```powershell
.\tools\msix\build-msix.ps1
```

If you want to rebuild the PyInstaller executable first:

```powershell
.\tools\msix\build-msix.ps1 -RunPyInstaller
```

The package is written to `dist\msix`.

## Partner Center identity

Before final Store submission, replace the default identity values with the
values shown in Partner Center for the reserved app name:

```powershell
.\tools\msix\build-msix.ps1 `
  -PackageName "Your.PartnerCenter.PackageName" `
  -Publisher "CN=Your Partner Center Publisher"
```

The defaults are development placeholders:

- `PackageName`: `Shirochi.Glossarion`
- `Publisher`: `CN=Shirochi`
- `PublisherDisplayName`: `Shirochi`

## Local install testing

Windows requires MSIX packages to be signed before local installation. Store
submission does not need this, but local double-click testing does.

```powershell
.\tools\msix\build-msix.ps1 `
  -Sign `
  -CertificatePath C:\path\to\test-or-code-signing-cert.pfx `
  -CertificatePassword "pfx-password"
```

The certificate subject must match the manifest publisher. For example, if the
manifest uses `-Publisher "CN=Shirochi"`, the certificate subject must also be
`CN=Shirochi`.
