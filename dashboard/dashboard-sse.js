/**
 * StrAIght Shot Dashboard — Live SSE Client
 * ==========================================
 * Connects to GET /v1/dash/stream and replaces static mock data
 * in code.html with live per-token telemetry from the middleware shim.
 *
 * Drop this into code.html just before </body> and the dashboard
 * comes alive.
 *
 * API contract (DashboardTokenEvent JSON):
 * {
 *   event_type: "token" | "verdict" | "system",
 *   request_id: "abc123",
 *   probe_score: 0.947,
 *   jspace_score: 0.89,
 *   entropy_value: 2.81,
 *   aggregate: 0.82,
 *   verdict: "BLOCK" | "WARN" | "PASS",
 *   jspace_concepts: [{name, value, confidence}, ...],
 *   silent_tokens: [{token, score}, ...],
 *   entropy_level: "ELEVATED" | "NOMINAL" | "CRITICAL",
 *   jspace_separation: 2.4,
 *   layer_activations: [0.1, 0.2, ...],
 *   vram_percent: 85.0,
 *   tokens_per_second: 45.2,
 *   uptime_seconds: 152248.0,
 *   kl_divergence: 0.014,
 *   heat_matrix: [[0.1, 0.2, ...], ...],
 * }
 */

(function () {
  'use strict';

  const DASHBOARD_URL =
    window.STRAIGHTSHOT_DASHBOARD_URL || '/v1/dash/stream';

  // ── DOM references (cache after first lookup) ──
  const dom = {
    verdictText: null,
    verdictIcon: null,
    probeScore: null,
    probeLabel: null,
    probeConf: null,
    entropyLevel: null,
    entropyGauge: null,
    entropySparkline: null,
    jspaceSep: null,
    silentTokens: null,
    heatMatrix: null,
    layerActivation: null,
    reqVolume: null,
    threatDist: null,
    klDivergence: null,
    klLabel: null,
    vramBar: null,
    vramPct: null,
    uptime: null,
    eventsLog: null,
    requestId: null,
    requestEndpoint: null,
    requestTimeAgo: null,
    requestPayload: null,
    alertBanner: null,
  };

  // ── Sparkline data buffers ──
  let entropyHistory = [];
  let reqVolumeHistory = [10, 12, 8, 15, 22, 30, 18, 25, 20, 14, 18, 28];
  const MAX_HISTORY = 60;

  // ── Event source ──
  let eventSource = null;
  let reconnectTimer = null;
  let reconnectDelay = 1000;

  function connect() {
    if (eventSource) {
      eventSource.close();
    }

    eventSource = new EventSource(DASHBOARD_URL);

    eventSource.addEventListener('token', (e) => {
      try {
        const data = JSON.parse(e.data);
        updateDashboard(data);
      } catch (err) {
        console.warn('[StraightShot] Parse error:', err);
      }
    });

    eventSource.addEventListener('verdict', (e) => {
      try {
        const data = JSON.parse(e.data);
        flashVerdict(data.verdict);
        addEvent('BLOCK', `Req #${data.request_id} intercepted.`);
      } catch (err) {
        console.warn('[StraightShot] Verdict parse error:', err);
      }
    });

    eventSource.addEventListener('system', (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.message !== 'ping') {
          addEvent(data.level, data.message);
        }
      } catch (err) {
        // ignore ping parse failures
      }
    });

    eventSource.onopen = () => {
      console.log('[StraightShot] Dashboard connected');
      reconnectDelay = 1000;
      addEvent('INFO', 'Dashboard connected.');
    };

    eventSource.onerror = () => {
      console.warn('[StraightShot] Connection lost, reconnecting...');
      eventSource.close();
      scheduleReconnect();
    };
  }

  function scheduleReconnect() {
    reconnectDelay = Math.min(reconnectDelay * 2, 30000);
    clearTimeout(reconnectTimer);
    reconnectTimer = setTimeout(connect, reconnectDelay);
  }

  // ── DOM update functions ──
  function resolveDOM() {
    // Lazy DOM resolution — runs on first update
    if (!dom.verdictText) {
      dom.verdictText = document.querySelector('[class*="text-error"][class*="font-headline-md"]');
      dom.probeScore = document.querySelector('[class*="text-signal-amber"][class*="font-headline-lg"]');
      
      // Find probe label (JAILBREAK badge)
      const badges = document.querySelectorAll('[class*="bg-error-container"]');
      if (badges.length > 0) dom.probeLabel = badges[0];
      
      // Confidence text
      const confSpans = document.querySelectorAll('[class*="font-mono-label"][class*="text-on-surface-variant"]');
      confSpans.forEach(s => {
        if (s.textContent.startsWith('CONF:')) dom.probeConf = s;
      });
      
      // Entropy level
      const elevated = document.querySelector('[class*="text-signal-amber"][class*="font-headline-md"]');
      if (elevated) dom.entropyLevel = elevated;
      
      // J-space separation
      document.querySelectorAll('[class*="font-mono-label"]').forEach(s => {
        if (s.textContent.startsWith('SEP:')) dom.jspaceSep = s;
      });
      
      // Silent tokens
      dom.silentTokens = document.querySelectorAll('[class*="text-signal-amber"][class*="drop-shadow"]');
      
      // VRAM bar + text
      dom.vramBar = document.querySelector('[class*="w-\\[85%\\]"]');
      dom.vramPct = document.querySelector('[class*="text-primary"]');
      
      // Uptime
      const uptimeEls = document.querySelectorAll('[class*="text-primary"]');
      uptimeEls.forEach(el => {
        if (/^\d+:\d+:\d+$/.test(el.textContent.trim())) dom.uptime = el;
      });
      
      // KL divergence
      document.querySelectorAll('[class*="font-headline-md"]').forEach(el => {
        if (/^\d+\.\d+$/.test(el.textContent.trim())) dom.klDivergence = el;
      });
      document.querySelectorAll('[class*="text-success-teal"][class*="drop-shadow"]').forEach(el => {
        if (el.textContent.includes('NOMINAL')) dom.klLabel = el;
      });
      
      // Request header
      document.querySelectorAll('[class*="font-headline-md"][class*="text-signal-amber"]').forEach(el => {
        if (el.textContent.startsWith('Req #')) dom.requestId = el;
      });
      document.querySelectorAll('[class*="text-on-surface-variant/70"][class*="uppercase"]').forEach(el => {
        if (el.textContent.startsWith('POST')) dom.requestEndpoint = el;
      });
      document.querySelectorAll('[class*="text-on-surface-variant"][class*="rounded"]').forEach(el => {
        if (el.textContent.endsWith('ago')) dom.requestTimeAgo = el;
      });
      
      // Alert banner
      dom.alertBanner = document.querySelector('[class*="glow-active"]');
      
      // Events log
      dom.eventsLog = document.querySelector('[class*="max-h-\\[150px\\]"]');
    }
  }

  function updateDashboard(data) {
    resolveDOM();
    
    // Request header
    if (dom.requestId && data.request_id) {
      dom.requestId.textContent = `Req #${data.request_id}`;
    }
    if (dom.requestTimeAgo) {
      dom.requestTimeAgo.textContent = '0.3s ago';
    }

    // Probe score
    if (dom.probeScore && data.probe_score !== undefined) {
      dom.probeScore.textContent = data.probe_score.toFixed(3);
    }

    // Probe label (jailbreak, injection, etc.)
    if (dom.probeLabel && data.jspace_concepts && data.jspace_concepts.length > 0) {
      const topConcept = data.jspace_concepts[0];
      dom.probeLabel.textContent = topConcept.name.toUpperCase();
    }

    // Confidence
    if (dom.probeConf && data.jspace_concepts && data.jspace_concepts.length > 0) {
      dom.probeConf.textContent = `CONF: ${data.jspace_concepts[0].confidence}%`;
    }

    // Verdict
    if (dom.verdictText) {
      dom.verdictText.textContent = data.verdict || 'PASS';
      if (data.verdict === 'BLOCK') {
        dom.verdictText.className = dom.verdictText.className
          .replace(/text-\w+/g, 'text-error')
          .replace(/drop-shadow-\[[\w\d_()'",.\s]*\]/g, "drop-shadow-[0_0_8px_theme('colors.error/40')]");
      } else if (data.verdict === 'WARN') {
        dom.verdictText.className = dom.verdictText.className
          .replace(/text-\w+/g, 'text-signal-amber')
          .replace(/drop-shadow-\[[\w\d_()'",.\s]*\]/g, "drop-shadow-[0_0_8px_theme('colors.signal-amber/40')]");
      }
    }

    // Entropy level
    if (dom.entropyLevel && data.entropy_level) {
      dom.entropyLevel.textContent = data.entropy_level;
    }

    // J-space separation
    if (dom.jspaceSep && data.jspace_separation !== undefined) {
      dom.jspaceSep.textContent = `SEP: ${data.jspace_separation}x`;
    }

    // VRAM
    if (dom.vramBar && data.vram_percent !== undefined) {
      dom.vramBar.style.width = `${Math.round(data.vram_percent)}%`;
    }
    if (dom.vramPct && data.vram_percent !== undefined) {
      // Find the exact VRAM percentage span (not uptime)
      document.querySelectorAll('[class*="text-primary"]').forEach(el => {
        if (el.textContent.includes('%')) {
          el.textContent = `${Math.round(data.vram_percent)}%`;
        }
      });
    }

    // Uptime
    if (dom.uptime && data.uptime_seconds !== undefined) {
      const h = Math.floor(data.uptime_seconds / 3600);
      const m = Math.floor((data.uptime_seconds % 3600) / 60);
      const s = Math.floor(data.uptime_seconds % 60);
      dom.uptime.textContent = `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    }

    // KL Divergence
    if (dom.klDivergence && data.kl_divergence !== undefined) {
      dom.klDivergence.textContent = data.kl_divergence.toFixed(3);
    }

    // Add to events log
    if (data.verdict === 'BLOCK') {
      addEvent('BLOCK', `Req #${data.request_id} intercepted.`);
    } else if (data.entropy_level === 'ELEVATED') {
      addEvent('WARN', `Entropy spike detected.`);
    }
  }

  function flashVerdict(verdict) {
    if (!dom.alertBanner) resolveDOM();
    if (dom.alertBanner) {
      dom.alertBanner.classList.add('glow-active');
      setTimeout(() => dom.alertBanner?.classList.remove('glow-active'), 2000);
    }
  }

  function addEvent(level, message) {
    if (!dom.eventsLog) resolveDOM();
    if (!dom.eventsLog) return;

    const now = new Date();
    const time = now.toTimeString().split(' ')[0];
    
    const line = document.createElement('div');
    line.className = level === 'BLOCK'
      ? 'text-error drop-shadow-[0_0_2px_theme(\'colors.error/50\')]'
      : level === 'WARN'
      ? 'text-signal-amber drop-shadow-[0_0_2px_theme(\'colors.signal-amber/50\')]'
      : 'text-system-blue';
    
    line.innerHTML = `<span class="text-on-surface-variant">[${time}]</span> ${level}: ${message}`;
    
    // Prepend to log
    if (dom.eventsLog.firstChild) {
      dom.eventsLog.insertBefore(line, dom.eventsLog.firstChild);
    } else {
      dom.eventsLog.appendChild(line);
    }
    
    // Trim log to last 50 entries
    while (dom.eventsLog.children.length > 50) {
      dom.eventsLog.removeChild(dom.eventsLog.lastChild);
    }
  }

  // ── Startup ──
  console.log('[StraightShot] Dashboard SSE client loaded');
  connect();
})();
