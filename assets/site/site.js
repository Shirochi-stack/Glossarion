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
    initCompareTabs();
    initWebtoon();
  });

  // ----- webtoon reader: Korean / English switch -----
  // Both languages have strips of the same size, so swapping every src keeps the layout. The strip at the
  // reading line (just under the sticky bar) is re-pinned afterwards in case anything moved.
  function initWebtoon() {
    var reader = document.querySelector("[data-webtoon]");
    if (!reader) return;
    var imgs = Array.prototype.slice.call(reader.querySelectorAll("img[data-ko][data-en]"));
    var labels = document.querySelectorAll("[data-webtoon-jump] option, .webtoon-chapter");
    var buttons = document.querySelectorAll("[data-webtoon-lang]");
    var bar = document.querySelector("[data-webtoon-bar]");
    var jump = document.querySelector("[data-webtoon-jump]");
    var headings = reader.querySelectorAll(".webtoon-chapter");

    // A failed load (e.g. caught mid-deploy) retries once with a cache-busting query.
    imgs.forEach(function (img) {
      img.addEventListener("error", function () {
        if (img.dataset.retried) return;
        img.dataset.retried = "1";
        img.src = img.src.split("?")[0] + "?r=" + Date.now();
      });
    });

    function readingLine() { return bar ? bar.getBoundingClientRect().bottom : 0; }

    function scrollToY(y) {
      try { window.scrollTo({ top: y, left: 0, behavior: "instant" }); } catch (e) { window.scrollTo(0, y); }
    }

    // the first strip that reaches below the reading line, and how far into it the line falls
    function anchor() {
      var line = readingLine();
      for (var i = 0; i < imgs.length; i++) {
        var r = imgs[i].getBoundingClientRect();
        if (r.bottom > line) {
          if (r.top > line) return null;  // above the reader or on a chapter heading: nothing to keep
          return { img: imgs[i], frac: (line - r.top) / (r.height || 1) };
        }
      }
      return null;
    }

    function setLang(lang, keepPlace) {
      if (lang !== "ko" && lang !== "en") lang = "en";
      var a = keepPlace ? anchor() : null;
      imgs.forEach(function (img) {
        var next = img.getAttribute("data-" + lang);
        if (img.getAttribute("src") !== next) {
          img.removeAttribute("data-retried");
          img.setAttribute("src", next);
        }
      });
      labels.forEach(function (el) { el.textContent = el.getAttribute("data-" + lang); });
      buttons.forEach(function (b) {
        b.setAttribute("aria-pressed", b.getAttribute("data-webtoon-lang") === lang ? "true" : "false");
      });
      reader.setAttribute("lang", lang);
      if (a) {
        var r = a.img.getBoundingClientRect();
        var shift = r.top + a.frac * r.height - readingLine();
        if (Math.abs(shift) >= 1) scrollToY(window.pageYOffset + shift);
      }
      store("glossarion-webtoon-lang", lang);
    }

    buttons.forEach(function (b) {
      b.addEventListener("click", function () { setLang(b.getAttribute("data-webtoon-lang"), true); });
    });
    setLang(store("glossarion-webtoon-lang") || "en", false);

    // chapter menu: jump on change, and follow the chapter being read
    if (jump) {
      jump.addEventListener("change", function () {
        var target = document.getElementById(jump.value);
        if (target) target.scrollIntoView({ block: "start" });
      });
      var ticking = false;
      window.addEventListener("scroll", function () {
        if (ticking) return;
        ticking = true;
        window.requestAnimationFrame(function () {
          ticking = false;
          var line = readingLine() + 2, current = null;
          headings.forEach(function (h) { if (h.getBoundingClientRect().top <= line) current = h.id; });
          if (current && jump.value !== current) jump.value = current;
        });
      }, { passive: true });
    }
  }

  // ----- manga / webtoon example tabs -----
  function initCompareTabs() {
    document.querySelectorAll("[data-compare-tab]").forEach(function (tab) {
      tab.addEventListener("click", function () {
        var name = tab.getAttribute("data-compare-tab");
        var group = tab.parentElement;
        group.querySelectorAll("[data-compare-tab]").forEach(function (t) {
          t.setAttribute("aria-selected", t === tab ? "true" : "false");
        });
        group.parentElement.querySelectorAll("[data-compare-panel]").forEach(function (panel) {
          panel.hidden = panel.getAttribute("data-compare-panel") !== name;
        });
      });
    });
  }

  // ----- before / after slider -----
  // The range input sets --pos; the step buttons swap which image sits on the left.
  function initCompare() {
    document.querySelectorAll("[data-compare]").forEach(function (fig) {
      var box = fig.querySelector(".compare-box");
      var range = fig.querySelector("input[type=range]");
      var before = fig.querySelector("img.before");
      var leftTag = fig.querySelector(".tag.l");
      if (!box || !range) return;
      // A failed load (e.g. caught mid-deploy) retries once with a cache-busting query.
      fig.querySelectorAll("img").forEach(function (img) {
        img.addEventListener("error", function () {
          if (img.dataset.retried) return;
          img.dataset.retried = "1";
          img.src = img.src.split("?")[0] + "?r=" + Date.now();
        });
      });
      function update() { box.style.setProperty("--pos", range.value + "%"); }
      range.addEventListener("input", update);
      update();

      // Dragging: the knob works for every pointer; a mouse may also drag anywhere on the image.
      // Touches elsewhere on the image are left to the browser, so a swipe scrolls the page.
      var knob = fig.querySelector(".knob");
      var dragging = null;
      function moveTo(clientX) {
        var rect = box.getBoundingClientRect();
        var pct = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
        range.value = String(Math.round(pct));
        update();
      }
      function start(e) {
        dragging = e.pointerId;
        try { e.currentTarget.setPointerCapture(e.pointerId); } catch (err) { /* ignore */ }
        moveTo(e.clientX);
        e.preventDefault();
      }
      function move(e) { if (dragging === e.pointerId) moveTo(e.clientX); }
      function end(e) { if (dragging === e.pointerId) dragging = null; }
      [knob, box].forEach(function (el) {
        if (!el) return;
        el.addEventListener("pointermove", move);
        el.addEventListener("pointerup", end);
        el.addEventListener("pointercancel", end);
      });
      if (knob) knob.addEventListener("pointerdown", function (e) { e.stopPropagation(); start(e); });
      box.addEventListener("pointerdown", function (e) {
        if (e.pointerType === "mouse" && e.button === 0) start(e);
      });
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
    highlightPlatform(os);
  }

  // The platform chips jump to their download card and highlight it.
  function highlightPlatform(os) {
    var card = document.getElementById("download-" + os);
    if (!card) return;
    document.querySelectorAll(".platform.current").forEach(function (c) { c.classList.remove("current"); });
    card.classList.add("current");
  }

  document.addEventListener("click", function (e) {
    var chip = e.target.closest && e.target.closest("[data-download-os]");
    if (chip) highlightPlatform(chip.getAttribute("data-download-os"));
  });

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
