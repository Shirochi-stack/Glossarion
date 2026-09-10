(() => {
  if (location.origin !== "http://127.0.0.1:18874" || location.pathname !== "/connect") return;
  const nonce = location.hash.slice(1);
  if (!/^[A-Za-z0-9_-]{16,256}$/.test(nonce)) return;
  // Retain the one-time link until pairing succeeds so installation can reload
  // this page and let a newly installed content script connect automatically.
  const status = document.createElement("p");
  status.setAttribute("role", "status");
  status.textContent = "Connecting this browser profile to Glossarion…";
  document.body.appendChild(status);
  chrome.runtime.sendMessage({type: "autharena-pair", nonce}).then(result => {
    if (result?.ok) history.replaceState(null, "", location.pathname + location.search);
    window.postMessage({type: "arena-pair-result", ok: !!result?.ok,
      ...(result?.ok ? {account_id: result.account_id} : {error: result?.error || "Arena pairing failed."})}, location.origin);
    status.textContent = result?.ok
      ? `Arena ${String(result.account_id)} connected. Opening Arena sign-in…`
      : `Arena connection failed: ${result?.error || "Open Arena Login in Glossarion to try again."}`;
  }).catch(() => {
    window.postMessage({type: "arena-pair-result", ok: false,
      error: "Could not contact the Arena companion. Reload it and open a new Arena Login link."}, location.origin);
    status.textContent = "Could not contact the Arena companion. Reload the extension, then click Arena Login in Glossarion again.";
  });
})();
