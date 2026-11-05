// Glossarion site — theme toggle, mobile menu, OS detection and latest-release links.
(function () {
  "use strict";

  var REPO = "Shirochi-stack/Glossarion";
  var root = document.documentElement;

  function store(key, value) {
    try {
      if (value === undefined) return localStorage.getItem(key);
      localStorage.setItem(key, value);
    } catch (e) { return null; }
  }

  // ----- theme -----
  var saved = store("glossarion-theme");
  if (saved === "light" || saved === "dark") root.setAttribute("data-theme", saved);

  function currentTheme() {
    var t = root.getAttribute("data-theme");
    if (t) return t;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }

  document.addEventListener("DOMContentLoaded", function () {
    var toggle = document.querySelector("[data-theme-toggle]");
    if (toggle) {
      toggle.addEventListener("click", function () {
        var next = currentTheme() === "dark" ? "light" : "dark";
        root.setAttribute("data-theme", next);
        store("glossarion-theme", next);
      });
    }

    // ----- mobile menu -----
    var menuBtn = document.querySelector("[data-menu-toggle]");
    var mobileNav = document.querySelector(".mobile-nav");
    if (menuBtn && mobileNav) {
      menuBtn.addEventListener("click", function () {
        var open = mobileNav.classList.toggle("open");
        menuBtn.setAttribute("aria-expanded", open ? "true" : "false");
      });
      mobileNav.addEventListener("click", function (e) {
        if (e.target.tagName === "A") {
          mobileNav.classList.remove("open");
          menuBtn.setAttribute("aria-expanded", "false");
        }
      });
    }

    detectPlatform();
    loadRelease();
    initCompare();
  });

  // ----- before / after slider -----
  // The range input sets --pos; the step buttons swap which image sits on the left.
  function initCompare() {
    document.querySelectorAll("[data-compare]").forEach(function (fig) {
      var box = fig.querySelector(".compare-box");
      var range = fig.querySelector("input[type=range]");
      var before = fig.querySelector("img.before");
      var leftTag = fig.querySelector(".tag.l");
      if (!box || !range) return;
      function update() { box.style.setProperty("--pos", range.value + "%"); }
      range.addEventListener("input", update);
      update();
      fig.querySelectorAll("[data-left]").forEach(function (btn) {
        btn.addEventListener("click", function () {
          before.src = btn.getAttribute("data-left");
          if (leftTag) leftTag.textContent = btn.getAttribute("data-label");
          fig.querySelectorAll("[data-left]").forEach(function (b) {
            b.setAttribute("aria-pressed", b === btn ? "true" : "false");
          });
        });
      });
    });
  }

  // ----- platform detection -----
  var OS_NAMES = { windows: "Windows", macos: "macOS", linux: "Linux", android: "Android", ios: "iOS" };

  function guessOS() {
    var ua = navigator.userAgent || "";
    var platform = (navigator.userAgentData && navigator.userAgentData.platform) || navigator.platform || "";
    if (/android/i.test(ua)) return "android";
    if (/iphone|ipad|ipod/i.test(ua) || (/mac/i.test(platform) && navigator.maxTouchPoints > 1)) return "ios";
    if (/win/i.test(platform) || /windows/i.test(ua)) return "windows";
    if (/mac/i.test(platform) || /mac os/i.test(ua)) return "macos";
    if (/linux|x11|cros/i.test(platform + ua)) return "linux";
    return null;
  }

  function detectPlatform() {
    var os = guessOS();
    if (!os) return;
    document.querySelectorAll("[data-download-label]").forEach(function (el) {
      el.textContent = "Download for " + OS_NAMES[os];
    });
    document.querySelectorAll("[data-download-primary]").forEach(function (el) {
      el.setAttribute("href", "#download-" + os);
    });
    var card = document.getElementById("download-" + os);
    if (card) card.classList.add("current");
  }

  // ----- latest release -----
  // Each download link carries data-asset="<regex>"; the static href already points at
  // the latest release page, so the page still works when the API is unreachable.
  function formatSize(bytes) {
    if (!bytes) return "";
    var mb = bytes / (1024 * 1024);
    return mb >= 1024 ? (mb / 1024).toFixed(1) + " GB" : Math.round(mb) + " MB";
  }

  function applyRelease(rel) {
    if (!rel || !rel.tag_name) return;
    document.querySelectorAll("[data-release-tag]").forEach(function (el) {
      el.textContent = rel.tag_name;
    });
    document.querySelectorAll("[data-release-date]").forEach(function (el) {
      try {
        el.textContent = new Date(rel.published_at).toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
      } catch (e) { /* keep fallback text */ }
    });
    document.querySelectorAll("[data-release-shown]").forEach(function (el) { el.hidden = false; });
    var assets = rel.assets || [];
    document.querySelectorAll("a[data-asset]").forEach(function (link) {
      var re;
      try { re = new RegExp(link.getAttribute("data-asset"), "i"); } catch (e) { return; }
      for (var i = 0; i < assets.length; i++) {
        if (re.test(assets[i].name)) {
          link.setAttribute("href", assets[i].browser_download_url);
          link.setAttribute("title", assets[i].name);
          var size = link.querySelector("span");
          if (size) size.textContent = formatSize(assets[i].size);
          return;
        }
      }
    });
  }

  function loadRelease() {
    if (!document.querySelector("a[data-asset], [data-release-tag]")) return;
    var cacheKey = "glossarion-latest-release";
    var cached = null;
    try { cached = JSON.parse(sessionStorage.getItem(cacheKey) || "null"); } catch (e) { cached = null; }
    if (cached && cached.tag_name) { applyRelease(cached); return; }
    if (!window.fetch) return;
    fetch("https://api.github.com/repos/" + REPO + "/releases/latest", { headers: { Accept: "application/vnd.github+json" } })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (rel) {
        if (!rel) return;
        var slim = {
          tag_name: rel.tag_name,
          published_at: rel.published_at,
          assets: (rel.assets || []).map(function (a) {
            return { name: a.name, size: a.size, browser_download_url: a.browser_download_url };
          })
        };
        try { sessionStorage.setItem(cacheKey, JSON.stringify(slim)); } catch (e) { /* ignore */ }
        applyRelease(slim);
      })
      .catch(function () { /* links already point at the releases page */ });
  }
})();
