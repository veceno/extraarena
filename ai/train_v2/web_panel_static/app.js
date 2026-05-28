/**
 * TrainV2 Web Panel Frontend
 * Vanilla JS, no external dependencies.
 */

const fmt = (v, digits = 3) => (v === null || v === undefined) ? "N/A" : (typeof v === "number" ? v.toFixed(digits) : v);
const fmtInt = (v) => (v === null || v === undefined) ? "N/A" : String(Math.round(v));

let currentData = null;

async function loadData() {
  try {
    const res = await fetch("/api/summary");
    currentData = await res.json();
    renderSummary(currentData);
    renderRuns(currentData.run_index);
    renderProfiles(currentData.profile_registry);
    renderReleases(currentData.release_bundles);
    renderEvidence(currentData.acceptance_reports, currentData.shadow_packs);
    renderArtifacts(currentData.artifacts);
    renderDoctor(currentData);
  } catch (err) {
    document.querySelector("#summary").textContent = "Error loading data: " + err.message;
  }
}

function renderSummary(data) {
  const el = document.querySelector("#summary");
  const lc = data.lifecycle || {};
  el.innerHTML = `
    <span class="badge">Runs: ${fmtInt(lc.runs)}</span>
    <span class="badge ${lc.profiles_errors > 0 ? 'warn' : ''}">Profiles OK/Err: ${fmtInt(lc.profiles_ok)}/${fmtInt(lc.profiles_errors)}</span>
    <span class="badge">Releases: ${fmtInt(lc.release_bundles)}</span>
    <span class="badge">Acceptance: ${fmtInt(lc.acceptance_pass + lc.acceptance_warn + lc.acceptance_fail)}</span>
    <span class="badge">Shadow Packs: ${fmtInt(lc.shadow_packs)}</span>
    ${lc.best_profile ? `<span class="badge">Best Score: ${fmt(lc.best_profile.score)}</span>` : ''}
  `;
}

function renderRuns(runIndex) {
  const rows = (runIndex && runIndex.rows) || [];
  const thead = document.querySelector("#table-runs thead");
  const tbody = document.querySelector("#table-runs tbody");
  thead.innerHTML = `
    <tr>
      <th>Name</th><th>Status</th><th>Updates</th><th>Steps</th>
      <th>Loss</th><th>WR Rand</th><th>WR ET</th><th>WR Greedy</th>
      <th>ONNX</th><th>Skipped</th>
    </tr>`;
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="10" class="empty">No runs found.</td></tr>`;
    return;
  }
  tbody.innerHTML = rows.map(r => `
    <tr data-dir="${encodeURIComponent(r.run_dir)}">
      <td>${r.name || "—"}</td>
      <td><span class="tag ${r.status === "ok" ? "ok" : "partial"}">${r.status || "—"}</span></td>
      <td>${fmtInt(r.updates)}</td>
      <td>${fmtInt(r.steps)}</td>
      <td>${fmt(r.last_loss)}</td>
      <td>${fmt(r.wr_random)}</td>
      <td>${fmt(r.wr_end_turn)}</td>
      <td>${fmt(r.wr_greedy_face)}</td>
      <td>${r.has_onnx ? "yes" : "no"}</td>
      <td>${fmtInt(r.skipped_updates)}</td>
    </tr>
  `).join("");
  tbody.querySelectorAll("tr").forEach(tr => {
    tr.addEventListener("click", () => showRunReport(decodeURIComponent(tr.dataset.dir)));
  });
}

async function showRunReport(dir) {
  const detail = document.querySelector("#detail-runs");
  detail.innerHTML = `<div class="detail-header"><span class="detail-title">Run Report</span><button class="btn-copy" data-path="${escapeHtml(dir)}">Copy path</button></div><div class="detail-body">Loading…</div>`;
  attachCopyButtons(detail);
  try {
    const res = await fetch("/api/run-report?dir=" + encodeURIComponent(dir));
    const data = await res.json();
    const body = detail.querySelector(".detail-body");
    if (data.error) {
      body.textContent = "Error: " + data.error;
      return;
    }
    const lines = [
      `## ${data.run_dir || dir}`,
      "",
      "### Summary",
      "- Updates: " + fmtInt((data.summary || {}).updates),
      "- Steps: " + fmtInt((data.summary || {}).steps),
      "- ONNX models: " + ((data.onnx_models || []).length),
      "",
      "### Metrics",
      "```json",
      JSON.stringify(data.metrics_summary || {}, null, 2),
      "```",
    ];
    body.innerHTML = "<pre>" + escapeHtml(lines.join("\n")) + "</pre>";
  } catch (err) {
    detail.querySelector(".detail-body").textContent = "Error: " + err.message;
  }
}

function renderProfiles(registry) {
  const rows = (registry && registry.rows) || [];
  const thead = document.querySelector("#table-profiles thead");
  const tbody = document.querySelector("#table-profiles tbody");
  thead.innerHTML = `
    <tr>
      <th>Rank</th><th>Model</th><th>Score</th><th>ONNX Exists</th>
      <th>Selection</th><th>Difficulty</th><th>Status</th>
    </tr>`;
  if (!rows.length) {
    tbody.innerHTML = `<tr><td colspan="7" class="empty">No profiles found.</td></tr>`;
    return;
  }
  tbody.innerHTML = rows.map((r, idx) => `
    <tr>
      <td>${r.status === "ok" ? (idx + 1) : "—"}</td>
      <td>${r.model_name || "—"}</td>
      <td>${fmt(r.score)}</td>
      <td>${r.onnx_exists ? "yes" : "no"}</td>
      <td>${r.selection || "—"}</td>
      <td>${r.difficulty || "—"}</td>
      <td><span class="tag ${r.status === "ok" ? "ok" : "error"}">${r.status || "—"}</span></td>
    </tr>
  `).join("");
}

function renderReleases(bundles) {
  const thead = document.querySelector("#table-releases thead");
  const tbody = document.querySelector("#table-releases tbody");
  thead.innerHTML = `
    <tr>
      <th>Model</th><th>Created</th><th>Missing</th><th>Files</th><th>Archive</th>
    </tr>`;
  if (!bundles.length) {
    tbody.innerHTML = `<tr><td colspan="5" class="empty">No release bundles found.</td></tr>`;
    return;
  }
  tbody.innerHTML = bundles.map(b => `
    <tr data-dir="${encodeURIComponent(b.bundle_dir)}">
      <td>${b.model_name || "—"}</td>
      <td>${b.created_at ? b.created_at.replace("T", " ").slice(0, 19) : "—"}</td>
      <td>${(b.missing || []).length}</td>
      <td>${fmtInt(b.files_count)}</td>
      <td>${b.archive_exists ? "yes" : "no"}</td>
    </tr>
  `).join("");
  tbody.querySelectorAll("tr").forEach(tr => {
    tr.addEventListener("click", () => {
      const dir = decodeURIComponent(tr.dataset.dir);
      showFileDetail("#detail-releases", "Release README", dir + "/README.md");
    });
  });
}

function renderEvidence(reports, packs) {
  // Acceptance
  const theadAg = document.querySelector("#table-acceptance thead");
  const tbodyAg = document.querySelector("#table-acceptance tbody");
  theadAg.innerHTML = `<tr><th>Status</th><th>Score</th><th>Path</th></tr>`;
  if (!reports.length) {
    tbodyAg.innerHTML = `<tr><td colspan="3" class="empty">No acceptance reports found.</td></tr>`;
  } else {
    tbodyAg.innerHTML = reports.map(r => `
      <tr data-dir="${encodeURIComponent(r.dir)}">
        <td><span class="tag ${r.status === "pass" ? "ok" : (r.status === "fail" ? "error" : "partial")}">${r.status || "—"}</span></td>
        <td>${fmt(r.score)}</td>
        <td class="path">${r.path || "—"}</td>
      </tr>
    `).join("");
    tbodyAg.querySelectorAll("tr").forEach(tr => {
      tr.addEventListener("click", () => {
        const dir = decodeURIComponent(tr.dataset.dir);
        showFileDetail("#detail-evidence", "Acceptance Gate", dir + "/acceptance_gate.md");
      });
    });
  }

  // Shadow packs
  const theadSh = document.querySelector("#table-shadow thead");
  const tbodySh = document.querySelector("#table-shadow tbody");
  theadSh.innerHTML = `<tr><th>Steps</th><th>Match Rate</th><th>Latency p95</th><th>Path</th></tr>`;
  if (!packs.length) {
    tbodySh.innerHTML = `<tr><td colspan="4" class="empty">No shadow packs found.</td></tr>`;
  } else {
    tbodySh.innerHTML = packs.map(p => `
      <tr data-dir="${encodeURIComponent(p.dir)}">
        <td>${fmtInt(p.steps)}</td>
        <td>${fmt(p.match_rate)}</td>
        <td>${fmt(p.overlay_latency_ms_p95, 1)} ms</td>
        <td class="path">${p.path || "—"}</td>
      </tr>
    `).join("");
    tbodySh.querySelectorAll("tr").forEach(tr => {
      tr.addEventListener("click", () => {
        const dir = decodeURIComponent(tr.dataset.dir);
        showFileDetail("#detail-evidence", "Shadow Summary", dir + "/shadow_summary.md");
      });
    });
  }
}

function renderArtifacts(artifacts) {
  const thead = document.querySelector("#table-artifacts thead");
  const tbody = document.querySelector("#table-artifacts tbody");
  thead.innerHTML = `<tr><th>Kind</th><th>Name</th><th>Status</th><th>Score</th><th>Created</th><th>Path</th></tr>`;
  if (!artifacts.length) {
    tbody.innerHTML = `<tr><td colspan="6" class="empty">No artifacts found.</td></tr>`;
    return;
  }
  tbody.innerHTML = artifacts.map(a => `
    <tr data-path="${encodeURIComponent(a.path || '')}">
      <td><span class="tag ${a.kind}">${a.kind}</span></td>
      <td>${a.name || "—"}</td>
      <td><span class="tag ${a.status === "ok" ? "ok" : (a.status === "error" ? "error" : "partial")}">${a.status || "—"}</span></td>
      <td>${fmt(a.score)}</td>
      <td>${a.created_at ? a.created_at.replace("T", " ").slice(0, 19) : "—"}</td>
      <td class="path">${a.display_path || a.path || "—"}</td>
    </tr>
  `).join("");
  tbody.querySelectorAll("tr").forEach(tr => {
    tr.addEventListener("click", () => {
      const path = decodeURIComponent(tr.dataset.path);
      if (path) {
        showArtifact(path);
      }
    });
  });
}

async function showArtifact(path) {
  const detail = document.querySelector("#detail-artifacts");
  detail.innerHTML = `<div class="detail-header"><span class="detail-title">Artifact</span><button class="btn-copy" data-path="${escapeHtml(path)}">Copy path</button></div><div class="detail-body">Loading…</div>`;
  attachCopyButtons(detail);
  try {
    const res = await fetch("/api/artifact?path=" + encodeURIComponent(path));
    const data = await res.json();
    const body = detail.querySelector(".detail-body");
    if (data.error) {
      body.textContent = "Error: " + data.error;
      return;
    }
    const content = data.content || "";
    if (data.content_type === "application/json") {
      body.innerHTML = "<pre>" + escapeHtml(JSON.stringify(JSON.parse(content), null, 2)) + "</pre>";
    } else {
      body.innerHTML = "<pre>" + escapeHtml(content) + "</pre>";
    }
  } catch (err) {
    detail.querySelector(".detail-body").textContent = "Error: " + err.message;
  }
}

async function showFileDetail(selector, title, path) {
  const detail = document.querySelector(selector);
  detail.innerHTML = `<div class="detail-header"><span class="detail-title">${escapeHtml(title)}</span><button class="btn-copy" data-path="${escapeHtml(path)}">Copy path</button></div><div class="detail-body">Loading…</div>`;
  attachCopyButtons(detail);
  try {
    const res = await fetch("/api/file?path=" + encodeURIComponent(path));
    const text = await res.text();
    const body = detail.querySelector(".detail-body");
    if (!res.ok) {
      body.textContent = "Error " + res.status + ": " + text;
      return;
    }
    body.innerHTML = "<pre>" + escapeHtml(text) + "</pre>";
  } catch (err) {
    detail.querySelector(".detail-body").textContent = "Error: " + err.message;
  }
}

function renderDoctor(data) {
  const el = document.querySelector("#doctor-content");
  const lc = data.lifecycle || {};
  const issues = [];
  if (lc.runs === 0) issues.push("No runs found");
  if (lc.profiles_ok === 0) issues.push("No profiles found");
  if (lc.profiles_errors > 0) issues.push("Profile registry has error rows");
  if (lc.release_bundles === 0) issues.push("No release bundles found");
  if ((lc.acceptance_pass + lc.acceptance_warn + lc.acceptance_fail) === 0) issues.push("No acceptance reports found");

  let html = `<h3>Doctor</h3>`;
  html += `<ul>`;
  html += `<li>Runs directory: ${data.runs_dir || "N/A"}</li>`;
  html += `<li>Releases directory: ${data.releases_dir || "N/A"}</li>`;
  html += `<li>Runs: ${fmtInt(lc.runs)}</li>`;
  html += `<li>Profiles OK/Errors: ${fmtInt(lc.profiles_ok)}/${fmtInt(lc.profiles_errors)}</li>`;
  html += `<li>Release bundles: ${fmtInt(lc.release_bundles)}</li>`;
  html += `<li>Acceptance pass/warn/fail: ${fmtInt(lc.acceptance_pass)}/${fmtInt(lc.acceptance_warn)}/${fmtInt(lc.acceptance_fail)}</li>`;
  html += `<li>Shadow packs: ${fmtInt(lc.shadow_packs)}</li>`;
  if (lc.best_profile) {
    html += `<li>Best profile: ${lc.best_profile.model_name || "unknown"} (score=${fmt(lc.best_profile.score)})</li>`;
  }
  html += `</ul>`;

  if (issues.length) {
    html += `<h4>Recommendations</h4><ul>`;
    for (const issue of issues) {
      html += `<li>${escapeHtml(issue)}</li>`;
    }
    html += `</ul>`;
  } else {
    html += `<p class="ok">No issues found.</p>`;
  }
  el.innerHTML = html;
}

function attachCopyButtons(container) {
  container.querySelectorAll(".btn-copy").forEach(btn => {
    btn.addEventListener("click", async (e) => {
      e.stopPropagation();
      const path = btn.dataset.path;
      try {
        if (navigator.clipboard) {
          await navigator.clipboard.writeText(path);
          btn.textContent = "Copied!";
          setTimeout(() => btn.textContent = "Copy path", 1200);
        } else {
          btn.textContent = "Unavailable";
        }
      } catch (err) {
        btn.textContent = "Failed";
      }
    });
  });
}

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, m => ({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[m]));
}

// Tabs + persistence
document.querySelectorAll(".tab-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    const tab = btn.dataset.tab;
    document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));
    document.querySelector("#panel-" + tab).classList.add("active");
    try { localStorage.setItem("trainv2_active_tab", tab); } catch (e) {}
  });
});

// Restore active tab
try {
  const saved = localStorage.getItem("trainv2_active_tab");
  if (saved) {
    const btn = document.querySelector(`.tab-btn[data-tab="${saved}"]`);
    if (btn) btn.click();
  }
} catch (e) {}

// Refresh button
document.querySelector("#refresh-btn").addEventListener("click", () => loadData());

loadData();
