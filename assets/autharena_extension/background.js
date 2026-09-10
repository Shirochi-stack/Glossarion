/* Arena credentials stay in the browser. Only status and generated output cross the bridge. */
"use strict";
try { importScripts("arena_page.js"); } catch (_) { /* Pairing can still report a missing generated helper. */ }

const BROKER = "http://127.0.0.1:18874";
const ARENA = "https://arena.ai";
const ALARM = "autharena-reconnect";
const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
let device = null;
let activeJob = null;
let ownedTab = null;
let polling = false;
let pairing = null;
let seenJobs = [];
let pendingErrors = [];

async function request(path, {body, authenticated = true, timeout = 35000} = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const headers = {};
    if (authenticated) {
      if (!device?.device_token) throw Error("Arena companion is not paired.");
      headers.Authorization = `Bearer ${device.device_token}`;
    }
    if (body !== undefined) headers["Content-Type"] = "application/json";
    const response = await fetch(BROKER + path, {
      method: body === undefined ? "GET" : "POST", headers,
      ...(body === undefined ? {} : {body: JSON.stringify(body)}),
      signal: controller.signal, credentials: "omit", redirect: "error", cache: "no-store"
    });
    if (!response.ok) {
      const error = Error(`Glossarion bridge returned HTTP ${response.status}.`);
      error.status = response.status;
      throw error;
    }
    return response.status === 204 ? {} : await response.json();
  } finally { clearTimeout(timer); }
}

function validConnectSender(sender) {
  try {
    const url = new URL(sender.url || "");
    return sender.tab?.id !== undefined && url.origin === BROKER && url.pathname === "/connect";
  } catch (_) { return false; }
}

async function pair(nonce) {
  if (!/^[A-Za-z0-9_-]{16,256}$/.test(nonce || "")) throw Error("Invalid pairing link.");
  await initializationPromise;
  const stored = await chrome.storage.local.get("autharenaDevice");
  const deviceId = stored.autharenaDevice?.device_id || crypto.randomUUID();
  const result = await request("/pair", {authenticated: false, timeout: 10000,
    body: {nonce, device_id: deviceId}});
  if (result.device_id !== deviceId || typeof result.device_token !== "string" || !result.device_token
      || !Number.isInteger(result.account_id) || result.account_id < 0 || result.account_id > 9999) {
    throw Error("Invalid response from the Glossarion pairing service.");
  }
  device = {device_id: deviceId, device_token: result.device_token, account_id: result.account_id};
  await chrome.storage.local.set({autharenaDevice: device});
  void startPolling();
  return {ok: true, account_id: device.account_id};
}

chrome.runtime.onMessage.addListener((message, sender, respond) => {
  if (message?.type !== "autharena-pair" || !validConnectSender(sender)) return false;
  if (pairing) { respond({ok: false, error: "Pairing is already in progress."}); return false; }
  pairing = pair(message.nonce);
  pairing.then(respond, error => respond({ok: false, error: error?.status === 409
    ? "This browser profile is already paired with another Arena slot. Use a different browser profile for another slot."
    : "Pairing failed. Open a new Arena Login link in Glossarion."}))
    .finally(() => { pairing = null; });
  return true;
});

async function rememberJob(job) {
  await chrome.storage.local.set({autharenaActiveJob: job ? {
    id: job.id, tabId: job.tabId, dispatched: job.dispatched
  } : null});
}

async function postEvents(jobId, events) {
  if (events.length) await request("/extension/events", {timeout: 10000, body: {job_id: jobId, events}});
}

async function execute(tabId, func, args = []) {
  const results = await chrome.scripting.executeScript({target: {tabId}, world: "MAIN", func, args});
  return results?.[0]?.result;
}

// These functions execute in the Arena document and have no extension privileges.
function pollArenaPage(jobId) {
  if (location.origin !== "https://arena.ai") return {arena: false};
  const state = window.__glossarionArena;
  const ours = state?.glossarionJobId === jobId;
  return {arena: true, loaded: document.readyState !== "loading", document: String(performance.timeOrigin),
    phase: ours ? state.phase : null, rejections: ours ? state.rejections || 0 : 0,
    events: ours ? state.events.splice(0, 64) : []};
}

function tagArenaJob(jobId) {
  if (window.__glossarionArena) window.__glossarionArena.glossarionJobId = jobId;
}

function dispatchArenaJob(jobId) {
  const state = window.__glossarionArena;
  if (state?.glossarionJobId !== jobId || state.phase !== "ready") return false;
  void state.dispatch();
  return true;
}

function abortArenaJob(jobId) {
  const state = window.__glossarionArena;
  if (state?.glossarionJobId === jobId) {
    state.controller?.abort();
    state.phase = "cancelled";
    document.getElementById("glossarion-arena-captcha")?.remove();
  }
}

function openArenaLogin() {
  if (location.origin !== "https://arena.ai") return false;
  const visible = node => {
    const style = getComputedStyle(node);
    return node.getClientRects().length && style.visibility !== "hidden" && style.display !== "none";
  };
  // Once Arena has opened a login/terms dialog, all form choices and credential
  // submission belong to the user. Never click a dialog's own "Log In" submit.
  if ([...document.querySelectorAll('[role="dialog"], dialog[open]')].some(visible)) return true;
  for (const node of document.querySelectorAll("button, a, [role=button]")) {
    if (node.closest('form, [role="dialog"], dialog')) continue;
    const label = (node.textContent || node.getAttribute("aria-label") || "").trim().replace(/\s+/g, " ");
    if (!/^(log\s*in|sign\s*in)$/i.test(label) || node.disabled || node.getAttribute("aria-disabled") === "true") continue;
    if (!visible(node)) continue;
    node.click();
    return true;
  }
  return false;
}

async function acquireTab(config) {
  const url = config.login ? ARENA + "/" : ARENA + "/text/direct?model_a=" + encodeURIComponent(config.model);
  const interactive = config.login || config.allow_interactive !== false;
  if (ownedTab) {
    try {
      const tab = await chrome.tabs.get(ownedTab.id);
      // If the user navigated our tab to another chat/site, leave it alone.
      if (tab.url === ownedTab.url && !tab.pinned) {
        await chrome.tabs.update(tab.id, {url, active: interactive});
        ownedTab = {id: tab.id, url};
        await chrome.storage.session.set({autharenaOwnedTab: ownedTab});
        return tab.id;
      }
    } catch (_) { /* Closed tabs are replaced only before a new job starts. */ }
  }
  const tab = await chrome.tabs.create({url, active: interactive});
  ownedTab = {id: tab.id, url};
  await chrome.storage.session.set({autharenaOwnedTab: ownedTab});
  return tab.id;
}

async function abortJob(job) {
  job.cancelled = true;
  if (job.tabId !== undefined) {
    try { await execute(job.tabId, abortArenaJob, [job.id]); } catch (_) {}
  }
}

async function failJob(job, message, bridgeLost = false,
                       errorType = /cancelled|canceled/i.test(message) ? "cancelled" : "api") {
  if (job.finished) return;
  job.finished = true;
  await abortJob(job);
  const event = {event: "error", message, error_type: errorType,
    request_dispatched: !!job.dispatched, safe_to_rotate: false};
  try {
    if (bridgeLost) throw Error("Bridge unavailable");
    await postEvents(job.id, [event]);
  } catch (_) {
    pendingErrors.push({job_id: job.id, events: [event]});
    await chrome.storage.local.set({autharenaPendingErrors: pendingErrors});
  }
  if (activeJob === job) activeJob = null;
  await rememberJob(null);
}

async function controlJob(job) {
  const config = job.config;
  const loginOnly = !!config.login;
  const interactive = loginOnly || config.allow_interactive !== false;
  let checkingLogin = loginOnly;
  let pendingLoginProbe = false;
  let installed = false;
  let documentId = null;
  let rejections = 0;
  let nextProbe = 0;
  let nextLoginClick = 0;
  try {
    if (typeof prepareArena !== "function") throw Error("Arena page helper is missing. Reinstall the companion from Glossarion.");
    job.tabId = await acquireTab(config);
    await rememberJob(job);
    while (!job.finished && !job.cancelled && Date.now() < job.deadline) {
      let state;
      try { state = await execute(job.tabId, pollArenaPage, [job.id]); }
      catch (error) {
        if (job.dispatched) throw Error("Arena browser context was lost after dispatch; completion is uncertain and the request was not retried.");
        // A user-closed tab is terminal; OAuth navigation before dispatch is expected.
        try { await chrome.tabs.get(job.tabId); }
        catch (_) { throw Error("Arena browser request cancelled because its tab was closed."); }
        await wait(150);
        continue;
      }
      if (!state?.arena) {
        if (job.dispatched) throw Error("Arena navigated after dispatch; completion is uncertain and the request was not retried.");
        await wait(150);
        continue;
      }
      if (documentId !== state.document) {
        if (job.dispatched) throw Error("Arena reloaded after dispatch; completion is uncertain and the request was not retried.");
        documentId = state.document;
        installed = false;
        pendingLoginProbe = false;
      }
      const forwarded = [];
      let terminal = false;
      for (const event of state.events || []) {
        const kind = event?.event;
        if (kind === "action") {
          if (interactive) await chrome.tabs.update(job.tabId, {active: true});
          if (state.phase === "waiting") { pendingLoginProbe = true; nextProbe = Date.now() + 2000; }
        }
        if (kind === "rejected") {
          job.dispatched = false;
          job.ready = false;
          rejections = Math.max(rejections, state.rejections || 0);
          await rememberJob(job);
        }
        if (kind === "done" && checkingLogin && !loginOnly) {
          checkingLogin = false; installed = false; pendingLoginProbe = false;
          continue;
        }
        if (kind === "ready") job.ready = true;
        if (["verified", "logged_out", "action", "ready", "rejected", "chunk", "done", "error", "status"].includes(kind)) {
          forwarded.push(kind === "error" ? {...event, request_dispatched: !!job.dispatched} : event);
        }
        if (kind === "done" || kind === "error") { terminal = true; break; }
      }
      await postEvents(job.id, forwarded);
      if (terminal) {
        job.finished = true;
        if (activeJob === job) activeJob = null;
        await rememberJob(null);
        return;
      }
      if (job.cancelled || job.finished) return;
      if (installed && state.phase === null) {
        if (job.dispatched) throw Error("Arena request state was lost after dispatch; the request was not retried.");
        installed = false;
      }
      if (interactive && (loginOnly || pendingLoginProbe) && Date.now() >= nextLoginClick) {
        await execute(job.tabId, openArenaLogin);
        nextLoginClick = Date.now() + 1500;
      }
      if (pendingLoginProbe && Date.now() >= nextProbe && state.phase === "waiting") {
        checkingLogin = true; installed = false; pendingLoginProbe = false;
      }
      if (!installed && state.loaded) {
        const pageConfig = {payload: config.payload, model: config.model || "",
          timeout_ms: Math.max(1, job.deadline - Date.now()), rejections, login_only: checkingLogin,
          allow_interactive: interactive, recaptcha_v2_sitekey: config.recaptcha_v2_sitekey};
        await execute(job.tabId, prepareArena, [pageConfig]);
        await execute(job.tabId, tagArenaJob, [job.id]);
        installed = true;
      }
      await wait(100);
    }
    if (!job.finished) await failJob(job,
      job.cancelled ? "Arena browser request cancelled." : "Arena request timed out waiting for the current browser.",
      false, job.cancelled ? "cancelled" : "timeout");
  } catch (error) {
    await failJob(job, error?.message || "Arena browser request failed.");
  }
}

async function acceptCommand(command) {
  if (!command || typeof command.job_id !== "string" || !command.job_id) return;
  if (command.type === "run") {
    if (seenJobs.includes(command.job_id)) return;
    seenJobs.push(command.job_id);
    seenJobs = seenJobs.slice(-1000);
    await chrome.storage.local.set({autharenaSeenJobs: seenJobs});
    if (activeJob) {
      await postEvents(command.job_id, [{event: "error", message: "This browser profile is already handling an Arena request.",
        request_dispatched: false, safe_to_rotate: true}]);
      return;
    }
    const config = command.config || {};
    const timeout = Number(config.timeout || 180);
    if (!Number.isFinite(timeout) || timeout <= 0 || timeout > 86400 || (!config.login && typeof config.model !== "string")) {
      await postEvents(command.job_id, [{event: "error", message: "Invalid Arena job configuration.", request_dispatched: false}]);
      return;
    }
    const job = {id: command.job_id, config, deadline: Date.now() + timeout * 1000,
      dispatched: false, ready: false, finished: false, cancelled: false};
    activeJob = job;
    await rememberJob(job);
    void controlJob(job);
    return;
  }
  const job = activeJob;
  if (!job || job.id !== command.job_id || job.finished) return;
  if (command.type === "cancel") {
    await failJob(job, "Arena browser request cancelled.");
  } else if (command.type === "dispatch") {
    if (job.dispatched) return; // A duplicate dispatch command must not send again.
    if (!job.ready || job.config.login) { await failJob(job, "Arena received an unexpected dispatch command."); return; }
    job.dispatched = true; job.ready = false;
    await rememberJob(job); // Persist uncertainty before crossing the POST boundary.
    try {
      const sent = await execute(job.tabId, dispatchArenaJob, [job.id]);
      if (!sent) { job.dispatched = false; throw Error("Arena was no longer ready to send."); }
    } catch (error) { await failJob(job, error?.message || "Arena dispatch failed; completion is uncertain."); }
  }
}

async function startPolling() {
  await initializationPromise;
  if (polling || !device?.device_token) return;
  polling = true;
  try {
    while (device?.device_token) {
      while (pendingErrors.length) {
        const pending = pendingErrors[0];
        try { await postEvents(pending.job_id, pending.events); }
        catch (error) {
          if (![404, 410].includes(error?.status)) throw error;
          // A restarted desktop may have expired this terminal job. Its ID
          // remains consumed locally even after the error is acknowledged here.
        }
        pendingErrors.shift();
        await chrome.storage.local.set({autharenaPendingErrors: pendingErrors});
      }
      const reply = await request("/extension/poll");
      await acceptCommand(reply.command);
      if (!reply.command) await wait(200);
    }
  } catch (error) {
    if (activeJob) await failJob(activeJob, "Arena desktop connection was lost; the request was cancelled and was not replayed.", true);
    if (error?.status === 401 || error?.status === 403) {
      device = device ? {device_id: device.device_id} : null;
      await chrome.storage.local.set({autharenaDevice: device});
    }
    // The alarm reconnects; delivered run IDs remain consumed.
  } finally { polling = false; }
}

chrome.tabs.onRemoved.addListener(tabId => {
  if (activeJob?.tabId === tabId) void failJob(activeJob, "Arena browser tab closed; request cancelled by user.");
  if (ownedTab?.id === tabId) ownedTab = null;
});
chrome.alarms.onAlarm.addListener(alarm => { if (alarm.name === ALARM) void startPolling(); });
chrome.runtime.onStartup.addListener(() => { void startPolling(); });
chrome.runtime.onInstalled.addListener(() => { void chrome.alarms.create(ALARM, {periodInMinutes: 0.5}); });

async function initialize() {
  const local = await chrome.storage.local.get(["autharenaDevice", "autharenaSeenJobs", "autharenaActiveJob", "autharenaPendingErrors"]);
  const session = await chrome.storage.session.get("autharenaOwnedTab");
  device = local.autharenaDevice || null;
  ownedTab = session.autharenaOwnedTab || null;
  seenJobs = local.autharenaSeenJobs || [];
  pendingErrors = local.autharenaPendingErrors || [];
  const interrupted = local.autharenaActiveJob;
  if (interrupted) {
    await abortJob(interrupted);
    pendingErrors.push({job_id: interrupted.id, events: [{event: "error",
      message: "Arena companion restarted during a request; it was cancelled and was not replayed.",
      request_dispatched: !!interrupted.dispatched, safe_to_rotate: false}]});
    if (!seenJobs.includes(interrupted.id)) seenJobs.push(interrupted.id);
    await chrome.storage.local.set({autharenaActiveJob: null, autharenaPendingErrors: pendingErrors, autharenaSeenJobs: seenJobs});
  }
  await chrome.alarms.create(ALARM, {periodInMinutes: 0.5});
}
const initializationPromise = initialize();
void startPolling();
