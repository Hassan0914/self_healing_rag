const API_BASE_URL = "https://selfhealingrag-production-4084.up.railway.app";

// DOM Elements
const backendStatus = document.getElementById("backendStatus");
const statusText = document.getElementById("statusText");
const statusDot = backendStatus.querySelector(".status-dot");

const corpusIdInput = document.getElementById("corpusId");
const checkCorpusBtn = document.getElementById("checkCorpusBtn");
const corpusStatusHelp = document.getElementById("corpusStatusHelp");

const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const fileNameDisplay = document.getElementById("fileNameDisplay");
const ingestForm = document.getElementById("ingestForm");
const ingestBtn = document.getElementById("ingestBtn");
const ingestResult = document.getElementById("ingestResult");

const topK = document.getElementById("topK");
const topKVal = document.getElementById("topKVal");
const maxRetries = document.getElementById("maxRetries");
const maxRetriesVal = document.getElementById("maxRetriesVal");

const askForm = document.getElementById("askForm");
const questionInput = document.getElementById("questionInput");
const askBtn = document.getElementById("askBtn");
const resultsContainer = document.getElementById("resultsContainer");

let selectedFile = null;

// Initialize
document.addEventListener("DOMContentLoaded", () => {
  checkBackendHealth();
  checkCorpusExistence();
  setupEventListeners();
});

// Event Listeners Setup
function setupEventListeners() {
  topK.addEventListener("input", (e) => topKVal.textContent = e.target.value);
  maxRetries.addEventListener("input", (e) => maxRetriesVal.textContent = e.target.value);

  checkCorpusBtn.addEventListener("click", checkCorpusExistence);
  corpusIdInput.addEventListener("change", checkCorpusExistence);

  // File Upload & Drag-Drop
  fileInput.addEventListener("change", handleFileSelect);

  dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  });

  dropZone.addEventListener("dragleave", () => {
    dropZone.classList.remove("dragover");
  });

  dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
    if (e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      handleFileSelect();
    }
  });

  ingestForm.addEventListener("submit", handleIngest);
  askForm.addEventListener("submit", handleAskQuestion);
}

// API: Check Backend Health
async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (res.ok) {
      statusDot.className = "status-dot online pulse";
      statusText.textContent = "API Live (Railway)";
      statusText.style.color = "#10b981";
    } else {
      throw new Error();
    }
  } catch {
    statusDot.className = "status-dot offline";
    statusText.textContent = "API Offline";
    statusText.style.color = "#ef4444";
  }
}

// API: Check Corpus Existence
async function checkCorpusExistence() {
  const corpusId = corpusIdInput.value.trim() || "default_corpus";
  corpusStatusHelp.textContent = "Checking corpus status...";
  corpusStatusHelp.style.color = "#94a3b8";

  try {
    const res = await fetch(`${API_BASE_URL}/corpora/${encodeURIComponent(corpusId)}/exists`);
    if (res.ok) {
      const data = await res.json();
      if (data.exists) {
        corpusStatusHelp.textContent = `✅ Corpus '${corpusId}' has ingested data.`;
        corpusStatusHelp.style.color = "#10b981";
      } else {
        corpusStatusHelp.textContent = `⚠️ Corpus '${corpusId}' is empty. Upload a document below.`;
        corpusStatusHelp.style.color = "#f59e0b";
      }
    }
  } catch {
    corpusStatusHelp.textContent = "Failed to query corpus status.";
    corpusStatusHelp.style.color = "#ef4444";
  }
}

// File Selection
function handleFileSelect() {
  if (fileInput.files.length) {
    selectedFile = fileInput.files[0];
    fileNameDisplay.textContent = `📄 ${selectedFile.name} (${(selectedFile.size / 1024).toFixed(1)} KB)`;
    ingestBtn.disabled = false;
  }
}

// API: Ingest Document
async function handleIngest(e) {
  e.preventDefault();
  if (!selectedFile) return;

  const corpusId = corpusIdInput.value.trim() || "default_corpus";
  const formData = new FormData();
  formData.append("file", selectedFile);
  formData.append("corpus_id", corpusId);

  ingestBtn.disabled = true;
  ingestBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Processing & Chunking...`;
  ingestResult.textContent = "";

  try {
    const res = await fetch(`${API_BASE_URL}/ingest`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    if (res.ok) {
      ingestResult.innerHTML = `<div style="color: #10b981;">✅ Ingested ${data.num_chunks} chunks into '${data.corpus_id}'.</div>`;
      checkCorpusExistence();
    } else {
      ingestResult.innerHTML = `<div style="color: #ef4444;">❌ Ingestion failed: ${data.detail || "Error"}</div>`;
    }
  } catch (err) {
    ingestResult.innerHTML = `<div style="color: #ef4444;">❌ Network error: ${err.message}</div>`;
  } finally {
    ingestBtn.disabled = false;
    ingestBtn.innerHTML = `<i class="fa-solid fa-upload"></i> Ingest Document`;
  }
}

// API: Ask Question
async function handleAskQuestion(e) {
  e.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;

  const corpusId = corpusIdInput.value.trim() || "default_corpus";
  askBtn.disabled = true;
  askBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Self-Healing...`;

  try {
    const res = await fetch(`${API_BASE_URL}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question: question,
        corpus_id: corpusId,
        top_k: parseInt(topK.value),
        max_retries: parseInt(maxRetries.value),
      }),
    });

    const data = await res.json();

    if (res.ok) {
      renderResponseCard(data);
    } else {
      alert(`Query failed: ${data.detail || res.statusText}`);
    }
  } catch (err) {
    alert(`Network Error: ${err.message}`);
  } finally {
    askBtn.disabled = false;
    askBtn.innerHTML = `<i class="fa-solid fa-paper-plane"></i> Ask`;
  }
}

// Render Response Card
function renderResponseCard(data) {
  // Clear empty state if present
  const emptyState = resultsContainer.querySelector(".empty-state");
  if (emptyState) emptyState.remove();

  const isFallback = data.was_fallback;
  const attemptsCount = data.attempts ? data.attempts.length : data.attempts_made || 1;

  const cardHtml = `
    <div class="response-card">
      <div class="response-header">
        <div class="question-title">
          <i class="fa-solid fa-circle-question" style="color: var(--accent-blue);"></i>
          ${escapeHtml(data.question)}
        </div>
      </div>

      <div class="answer-box ${isFallback ? 'fallback' : 'grounded'}">
        <strong>${isFallback ? '🛡️ Fallback Response:' : '✅ Answer:'}</strong><br>
        ${escapeHtml(data.final_answer)}
      </div>

      <div class="attempts-trace">
        <h4><i class="fa-solid fa-route"></i> Execution & Self-Healing Trace (${attemptsCount} attempt(s))</h4>
        ${renderAttempts(data.attempts || [])}
      </div>
    </div>
  `;

  resultsContainer.insertAdjacentHTML("afterbegin", cardHtml);
}

function renderAttempts(attempts) {
  return attempts.map((att) => {
    const isAccepted = att.accepted;
    const crit = att.critique || {};
    const verdict = crit.verdict || att.critique_verdict || "hallucinated";
    const isGrounded = verdict === "grounded";

    const badgeClass = isGrounded ? "badge-grounded" : "badge-hallucinated";
    const badgeText = isGrounded ? "🟢 grounded" : "🔴 hallucinated";

    const llmScore = crit.llm_faithfulness_score !== undefined ? crit.llm_faithfulness_score.toFixed(2) : "-";
    const embedScore = crit.embedding_grounding_score !== undefined ? crit.embedding_grounding_score.toFixed(2) : "-";
    const combinedScore = (crit.combined_score !== undefined ? crit.combined_score : att.combined_faithfulness_score || 0).toFixed(2);

    const chunks = att.retrieved_chunks || [];

    return `
      <div class="attempt-step">
        <div class="attempt-header">
          <span class="attempt-num">Attempt ${att.attempt_number || att.attempt_index} ${isAccepted ? '— ✅ Accepted' : '— ❌ Rejected'}</span>
          <span class="badge ${badgeClass}">${badgeText}</span>
        </div>

        <div style="font-size: 0.85rem; color: var(--text-secondary); margin-bottom: 0.5rem;">
          <strong>Query Used:</strong> <code>${escapeHtml(att.query_used)}</code>
        </div>

        <div class="score-grid">
          <div class="score-card">
            <span class="score-label">LLM Faithfulness</span>
            <span class="score-value">${llmScore}</span>
          </div>
          <div class="score-card">
            <span class="score-label">Embedding Grounding</span>
            <span class="score-value">${embedScore}</span>
          </div>
          <div class="score-card">
            <span class="score-label">Combined Score</span>
            <span class="score-value">${combinedScore}</span>
          </div>
        </div>

        ${crit.reasoning ? `<div style="font-size: 0.82rem; color: var(--text-muted); margin-bottom: 0.4rem;"><em>Reasoning: ${escapeHtml(crit.reasoning)}</em></div>` : ''}

        <details>
          <summary>View ${chunks.length} Context Chunk(s)</summary>
          ${chunks.map((c) => `
            <div class="chunk-item">
              <span class="chunk-source">Source: ${escapeHtml(c.source || "unknown")} | Score/Dist: ${(c.distance || 0).toFixed(3)}</span>
              ${escapeHtml((c.text || "").substring(0, 300))}...
            </div>
          `).join("")}
        </details>
      </div>
    `;
  }).join("");
}

function escapeHtml(str) {
  return (str || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}
