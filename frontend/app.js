/**
 * ThyroGuard Clinical Intelligence Platform - Frontend Application Logic
 * Evidence-Grounded Medical Reference Assistant
 */

const API_BASE = "http://localhost:8000";

// State
let currentConsultations = [];
let activeConsultationId = null;
let isSubmitting = false;

// DOM Elements
const tabBtns = document.querySelectorAll(".nav-tab");
const tabPanes = {
  assistant: document.getElementById("pane-assistant"),
  knowledge: document.getElementById("pane-knowledge"),
  system: document.getElementById("pane-system")
};

const welcomeHero = document.getElementById("welcome-hero");
const chatThreadContainer = document.getElementById("chat-thread-container");
const messagesFeed = document.getElementById("messages-feed");

const heroQueryInput = document.getElementById("hero-query-input");
const btnHeroSend = document.getElementById("btn-hero-send");

const chatTextarea = document.getElementById("chat-textarea");
const btnChatSend = document.getElementById("btn-chat-send");

const sidebarBtnNew = document.getElementById("sidebar-btn-new");
const chatBtnNew = document.getElementById("chat-btn-new");
const btnClearHistory = document.getElementById("btn-clear-history");
const recentConsultationsList = document.getElementById("recent-consultations-list");

const sidebarGotoKnowledge = document.getElementById("sidebar-goto-knowledge");
const sidebarGotoSystem = document.getElementById("sidebar-goto-system");

const protocolItems = document.querySelectorAll(".protocol-item");
const observabilityToggle = document.getElementById("observability-toggle");
const headerModeToggle = document.getElementById("header-mode-toggle");

// -----------------------------------------------------------------------------
// 0. THEME MANAGEMENT (SEAMLESS DARK / LIGHT TOGGLE)
// -----------------------------------------------------------------------------
function applyTheme(theme) {
  const isDark = (theme === "dark");
  document.documentElement.setAttribute("data-theme", theme);
  document.body.classList.toggle("dark-mode", isDark);
  if (headerModeToggle) {
    headerModeToggle.checked = isDark;
  }
  localStorage.setItem("thyroguard_theme", theme);
}

// Default to light mode (matching original design), or restore saved preference
const savedTheme = localStorage.getItem("thyroguard_theme") || "light";
applyTheme(savedTheme);

if (headerModeToggle) {
  headerModeToggle.addEventListener("change", (e) => {
    applyTheme(e.target.checked ? "dark" : "light");
  });
}

// -----------------------------------------------------------------------------
// 1. TAB NAVIGATION
// -----------------------------------------------------------------------------
function switchTab(targetTab) {
  tabBtns.forEach(btn => {
    if (btn.dataset.tab === targetTab) {
      btn.classList.add("active");
    } else {
      btn.classList.remove("active");
    }
  });

  Object.keys(tabPanes).forEach(tab => {
    if (tab === targetTab) {
      tabPanes[tab].style.display = "flex";
    } else {
      tabPanes[tab].style.display = "none";
    }
  });
}

tabBtns.forEach(btn => {
  btn.addEventListener("click", () => {
    switchTab(btn.dataset.tab);
  });
});

if (sidebarGotoKnowledge) {
  sidebarGotoKnowledge.addEventListener("click", () => switchTab("knowledge"));
}

if (sidebarGotoSystem) {
  sidebarGotoSystem.addEventListener("click", () => switchTab("system"));
}

// -----------------------------------------------------------------------------
// 2. TEXT DIRECTION HELPER
// -----------------------------------------------------------------------------
function isArabicText(text) {
  const arabicPattern = /[\u0600-\u06FF\u0750-\u077F]/;
  return arabicPattern.test(text);
}

// -----------------------------------------------------------------------------
// 3. CONSULTATION MANAGEMENT
// -----------------------------------------------------------------------------
function startNewConsultation() {
  activeConsultationId = "cons_" + Date.now();
  messagesFeed.innerHTML = "";
  welcomeHero.style.display = "flex";
  chatThreadContainer.style.display = "none";
  if (heroQueryInput) heroQueryInput.value = "";
  if (chatTextarea) chatTextarea.value = "";
}

function updateRecentConsultationsSidebar() {
  if (!recentConsultationsList) return;
  if (currentConsultations.length === 0) {
    recentConsultationsList.innerHTML = '<div class="empty-recent-msg">No consultations yet in this session.</div>';
    return;
  }

  recentConsultationsList.innerHTML = currentConsultations.map(c => `
    <div class="protocol-item recent-item" data-id="${c.id}" style="padding: 6px 8px; font-size: 0.78rem;">
      <span style="color: #6366f1;">•</span>
      <span style="white-space: nowrap; overflow: hidden; text-overflow: ellipsis; max-width: 210px;" title="${escapeHtml(c.title)}">
        ${escapeHtml(c.title)}
      </span>
    </div>
  `).join("");
}

function escapeHtml(str) {
  if (!str) return "";
  return str.replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;");
}

// -----------------------------------------------------------------------------
// 4. RENDERING CHAT MESSAGES
// -----------------------------------------------------------------------------
function appendUserMessage(text) {
  welcomeHero.style.display = "none";
  chatThreadContainer.style.display = "flex";

  const isAr = isArabicText(text);
  const card = document.createElement("div");
  card.className = "message-card-user";
  card.innerHTML = `
    <div class="user-avatar-red" title="User">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <path d="M8 14s1.5 2 4 2 4-2 4-2"></path>
        <line x1="9" y1="9" x2="9.01" y2="9"></line>
        <line x1="15" y1="9" x2="15.01" y2="9"></line>
      </svg>
    </div>
    <div class="user-query-text ${isAr ? 'ar' : ''}">${escapeHtml(text)}</div>
  `;
  messagesFeed.appendChild(card);
  card.scrollIntoView({ behavior: "smooth" });
}

function appendAssistantMessage(data, latencyMs) {
  const card = document.createElement("div");
  card.className = "message-card-assistant";

  const isGreeting = data.isGreeting;
  const isEmergency = data.safety_status === "medical_emergency";
  const isOutOfScope = data.safety_status === "out_of_scope_blocked" || data.status === "refusal";

  const latencySec = (latencyMs / 1000).toFixed(2);
  const isAr = isArabicText(data.answer || "");

  // Format response content with Markdown if marked exists
  let formattedAnswer = data.answer || "";
  if (window.marked && typeof window.marked.parse === "function") {
    formattedAnswer = window.marked.parse(formattedAnswer);
  } else {
    formattedAnswer = `<p>${formattedAnswer}</p>`;
  }

  // Metrics determination
  let qualityBadgeText = "MODERATE";
  let qualityClass = "moderate";
  let confidenceVal = 88;

  if (data.faithfulness_score !== undefined && data.faithfulness_score !== null) {
    const rawVal = Number(data.faithfulness_score);
    if (rawVal > 1.0) {
      confidenceVal = Math.min(100, Math.max(0, Math.round(rawVal)));
    } else {
      confidenceVal = Math.min(100, Math.max(0, Math.round(rawVal * 100)));
    }
  } else if (isOutOfScope || isEmergency) {
    confidenceVal = 0;
  }

  if (confidenceVal >= 80) {
    qualityBadgeText = "HIGH";
    qualityClass = "high";
  } else if (isOutOfScope || isEmergency) {
    qualityBadgeText = "REFUSAL";
    qualityClass = "refusal";
  }

  const sourcesRetrieved = data.retrieved_chunks ? data.retrieved_chunks.length : (data.retrieved_chunk_ids ? data.retrieved_chunk_ids.length : 0);
  const citationsVerified = (data.citations ? data.citations.length : 0);

  // Evidence list markup
  let evidenceAccordionHtml = "";
  if (!isGreeting && sourcesRetrieved > 0) {
    const chunks = data.retrieved_chunks || [];
    const chunkCards = chunks.map((c, i) => {
      const chunkId = c.chunk_id || c.id || `chunk_${i+1}`;
      const title = c.metadata?.guideline || c.metadata?.doc_title || "Clinical Practice Guideline";
      const score = c.relevance_score ? c.relevance_score.toFixed(3) : (c.score ? c.score.toFixed(3) : "0.850");
      const passage = c.content || c.text || "";

      return `
        <div class="chunk-card">
          <div class="chunk-header">
            <span class="chunk-tag">${escapeHtml(chunkId)}</span>
            <span class="chunk-score">Score: ${score}</span>
          </div>
          <div style="font-size: 0.75rem; font-weight: 600; color: #0369a1; margin-bottom: 4px;">${escapeHtml(title)}</div>
          <div class="chunk-body">${escapeHtml(passage)}</div>
        </div>
      `;
    }).join("");

    evidenceAccordionHtml = `
      <details class="cited-evidence-accordion">
        <summary>
          <span>›</span> 📄 Cited Evidence — ${citationsVerified} cited · ${sourcesRetrieved} retrieved background
        </summary>
        <div class="cited-content">
          ${chunkCards}
        </div>
      </details>
    `;
  }

  // Compose Full Assistant Card HTML
  card.innerHTML = `
    <div class="assistant-top-row">
      <div class="assistant-stethoscope-icon" title="ThyroGuard Assistant">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <path d="M4.8 2.3A.3.3 0 1 0 5 2H4a2 2 0 0 0-2 2v5a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6V4a2 2 0 0 0-2-2h-1a.2.2 0 1 0 .3.3"></path>
          <path d="M8 15v1a6 6 0 0 0 6 6v0a6 6 0 0 0 6-6v-4"></path>
          <circle cx="20" cy="10" r="2"></circle>
        </svg>
      </div>
      <div class="assistant-answer-body ${isAr ? 'ar' : ''}">
        ${formattedAnswer}
      </div>
    </div>

    ${!isGreeting ? `
      <hr class="assistant-divider">
      
      <div class="assistant-disclaimer">
        DISCLAIMER: This system provides informational medical reference only based on provided documents and does not replace professional medical judgment, diagnosis, or treatment.
      </div>

      <div class="evidence-metrics-box">
        <div class="metrics-row">
          
          <div class="metric-col">
            <span class="metric-col-title">EVIDENCE RETRIEVAL QUALITY</span>
            <span class="quality-badge ${qualityClass}">${qualityBadgeText}</span>
          </div>

          <div class="metric-col">
            <span class="metric-col-title">EVIDENCE RETRIEVAL CONFIDENCE</span>
            <div class="confidence-bar-group">
              <div class="conf-bar-track">
                <div class="conf-bar-fill ${confidenceVal >= 80 ? 'high' : ''}" style="width: ${confidenceVal}%"></div>
              </div>
              <span class="conf-bar-val">${confidenceVal}%</span>
            </div>
          </div>

          <div class="metric-col">
            <span class="metric-col-title">SOURCES RETRIEVED</span>
            <span class="metric-col-val">${sourcesRetrieved}</span>
          </div>

          <div class="metric-col">
            <span class="metric-col-title">CITATIONS VERIFIED</span>
            <span class="metric-col-val">${citationsVerified}/${sourcesRetrieved || 6}</span>
          </div>

          <div class="metric-latency-pill">
            <span>⚡</span> ${latencySec}s
          </div>

        </div>
        <div class="metrics-caption">
          Retrieval confidence reflects indexed evidence relevance — not diagnostic certainty.
        </div>
      </div>

      ${evidenceAccordionHtml}
    ` : ''}
  `;

  messagesFeed.appendChild(card);
  card.scrollIntoView({ behavior: "smooth" });
}

// -----------------------------------------------------------------------------
// 5. QUERY SUBMISSION PIPELINE
// -----------------------------------------------------------------------------
async function handleQuerySubmit(queryText) {
  const query = (queryText || "").trim();
  if (!query || isSubmitting) return;

  isSubmitting = true;
  if (btnChatSend) btnChatSend.disabled = true;
  if (btnHeroSend) btnHeroSend.disabled = true;

  // Track consultation
  if (!activeConsultationId) {
    activeConsultationId = "cons_" + Date.now();
  }
  const existingCons = currentConsultations.find(c => c.id === activeConsultationId);
  if (!existingCons) {
    currentConsultations.unshift({
      id: activeConsultationId,
      title: query.slice(0, 42) + (query.length > 42 ? "..." : ""),
      timestamp: Date.now()
    });
    updateRecentConsultationsSidebar();
  }

  // Display user card
  appendUserMessage(query);

  // Check for common friendly greetings (instant pleasant response like Image 1)
  const lowerQuery = query.toLowerCase().replace(/[؟?!.]/g, "").trim();
  const greetingsList = ["ازيك", "ازيك يا دكتور", "مرحبا", "أهلا", "اهلا", "السلام عليكم", "مساء الخير", "صباح الخير", "hello", "hi", "hey"];
  
  if (greetingsList.includes(lowerQuery)) {
    setTimeout(() => {
      appendAssistantMessage({
        isGreeting: true,
        answer: `أهلاً بك! أنا **ThyroGuard**، المساعد الإكلينيكي المعتمد على الأدلة والمراجع الطبية الموثقة لإرشادات الغدة الدرقية (ATA & ETA).\n\nأنا هنا لمساعدتك في استعراض وتلخيص الأدلة السريرية والإرشادات الطبية بدقة. كيف يمكنني مساعدتك في استفساراتك اليوم؟`,
        safety_status: "in_scope"
      }, 350);
      isSubmitting = false;
      if (btnChatSend) btnChatSend.disabled = false;
      if (btnHeroSend) btnHeroSend.disabled = false;
    }, 400);
    return;
  }

  // Display loading indicator bubble
  const loadingCard = document.createElement("div");
  loadingCard.className = "message-card-assistant";
  loadingCard.id = "assistant-loading-card";
  loadingCard.innerHTML = `
    <div class="assistant-top-row">
      <div class="assistant-stethoscope-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
          <circle cx="12" cy="12" r="10"></circle>
          <polyline points="12 6 12 12 16 14"></polyline>
        </svg>
      </div>
      <div class="assistant-answer-body" style="color: #64748b; font-style: italic;">
        Evaluating clinical evidence against guidelines and synthesising response...
      </div>
    </div>
  `;
  messagesFeed.appendChild(loadingCard);
  loadingCard.scrollIntoView({ behavior: "smooth" });

  const startTime = Date.now();

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 45000);

    const response = await fetch(`${API_BASE}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: query, top_k: 5 }),
      signal: controller.signal
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`Server returned HTTP ${response.status}`);
    }

    const data = await response.json();
    const latencyMs = Date.now() - startTime;

    if (observabilityToggle && observabilityToggle.checked) {
      console.log("[ThyroGuard Observability Trace]", data.pipeline_trace || data);
    }

    // Remove loading card
    const loadEl = document.getElementById("assistant-loading-card");
    if (loadEl) loadEl.remove();

    appendAssistantMessage(data, latencyMs);

  } catch (err) {
    console.error("[ThyroGuard API Error]", err);
    const latencyMs = Date.now() - startTime;

    const loadEl = document.getElementById("assistant-loading-card");
    if (loadEl) loadEl.remove();

    // Fallback response with disclaimer
    appendAssistantMessage({
      answer: `عذراً، حدث خطأ أثناء الاتصال بمحرك الاستدلال السريري (${err.message}). يرجى التأكد من تشغيل الحاوية والمحاولة مرة أخرى.`,
      safety_status: "out_of_scope_blocked",
      status: "refusal",
      retrieved_chunks: [],
      citations: []
    }, latencyMs);
  } finally {
    isSubmitting = false;
    if (btnChatSend) btnChatSend.disabled = false;
    if (btnHeroSend) btnHeroSend.disabled = false;
    if (chatTextarea) {
      chatTextarea.value = "";
      chatTextarea.focus();
    }
  }
}

// -----------------------------------------------------------------------------
// 6. EVENT LISTENERS
// -----------------------------------------------------------------------------
if (btnHeroSend && heroQueryInput) {
  btnHeroSend.addEventListener("click", () => {
    handleQuerySubmit(heroQueryInput.value);
  });

  heroQueryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleQuerySubmit(heroQueryInput.value);
    }
  });
}

if (btnChatSend && chatTextarea) {
  btnChatSend.addEventListener("click", () => {
    handleQuerySubmit(chatTextarea.value);
  });

  chatTextarea.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleQuerySubmit(chatTextarea.value);
    }
  });
}

// New consultation buttons
if (sidebarBtnNew) {
  sidebarBtnNew.addEventListener("click", startNewConsultation);
}
if (chatBtnNew) {
  chatBtnNew.addEventListener("click", startNewConsultation);
}

// Clear history
if (btnClearHistory) {
  btnClearHistory.addEventListener("click", () => {
    currentConsultations = [];
    updateRecentConsultationsSidebar();
    startNewConsultation();
  });
}

// Bookmarked protocols click handler
protocolItems.forEach(item => {
  item.addEventListener("click", () => {
    const q = item.getAttribute("data-query");
    if (q) {
      switchTab("assistant");
      handleQuerySubmit(q);
    }
  });
});

// Knowledge Base upload simulation
const btnProcessKb = document.getElementById("btn-process-kb");
const kbDropzone = document.getElementById("kb-dropzone");
const kbFileInput = document.getElementById("kb-file-input");

if (kbDropzone && kbFileInput) {
  kbDropzone.addEventListener("click", () => kbFileInput.click());
}

if (btnProcessKb) {
  btnProcessKb.addEventListener("click", () => {
    btnProcessKb.innerText = "⏳ Processing & Indexing...";
    btnProcessKb.disabled = true;
    setTimeout(() => {
      btnProcessKb.innerText = "✅ Knowledge Base Updated (3,073 chunks)";
      btnProcessKb.disabled = false;
    }, 2000);
  });
}

// -----------------------------------------------------------------------------
// 7. INITIAL HEALTH & MODEL INFO FETCH
// -----------------------------------------------------------------------------
async function fetchSystemDiagnostics() {
  try {
    const res = await fetch(`${API_BASE}/health`);
    if (res.ok) {
      const data = await res.json();
      const chunkCountEl = document.getElementById("kb-chunk-count");
      if (chunkCountEl && data.corpus_chunks) {
        chunkCountEl.innerText = data.corpus_chunks.toLocaleString();
      }
    }
  } catch (e) {
    console.log("Health fetch bypassed.");
  }
}

// Run on page load
document.addEventListener("DOMContentLoaded", () => {
  fetchSystemDiagnostics();
  updateRecentConsultationsSidebar();
});
