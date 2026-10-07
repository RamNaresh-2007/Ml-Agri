let currentPage = 1;
let totalPages = 1;
let currentSearch = "";
let yieldChartInstance = null;
let seasonPieInstance = null;
let topCropsInstance = null;
let topStatesInstance = null;
let sensitivityChartInstance = null;
let featureImportanceInstance = null;
let cachedDashboardData = null;
let cachedFeatureData = null;

// Theme color definitions for Chart.js
const THEME_PALETTES = {
    'midnight-cyber': {
        textColor: '#94a3b8',
        gridColor: 'rgba(255, 255, 255, 0.08)',
        barPrimary: '#8b5cf6',
        barSecondary: '#06b6d4'
    },
    'obsidian-emerald': {
        textColor: '#94a3b8',
        gridColor: 'rgba(16, 185, 129, 0.12)',
        barPrimary: '#10b981',
        barSecondary: '#06b6d4'
    },
    'sunset-nebula': {
        textColor: '#c4b5fd',
        gridColor: 'rgba(217, 70, 239, 0.12)',
        barPrimary: '#f43f5e',
        barSecondary: '#a855f7'
    },
    'solar-flare': {
        textColor: '#d4cfc7',
        gridColor: 'rgba(245, 158, 11, 0.12)',
        barPrimary: '#f59e0b',
        barSecondary: '#ea580c'
    },
    'nordic-frost': {
        textColor: '#475569',
        gridColor: '#e2e8f0',
        barPrimary: '#2563eb',
        barSecondary: '#0ea5e9'
    }
};

function getCurrentTheme() {
    return document.documentElement.getAttribute('data-theme') || 'obsidian-emerald';
}

function getPalette() {
    const t = getCurrentTheme();
    return THEME_PALETTES[t] || THEME_PALETTES['obsidian-emerald'];
}

document.addEventListener("DOMContentLoaded", () => {
    initThemeSwitcher();
    setupNavigation();
    loadDashboardData();
    loadMetadata();
    loadRecords(1);
    loadReportsGallery();
    setupReportHub();
    loadModelsComparison();
    loadFeatureImportances();
    setupPredictorForm();
    setupCropRecommender();
    setupDosageOptimizer();
    setupBatchSimulation();
    setupModal();
});

/* --------------------------------------------------------------------------
   Theme Switcher
   -------------------------------------------------------------------------- */
function initThemeSwitcher() {
    const swatchBtns = document.querySelectorAll(".theme-swatch-btn");
    const currentTheme = getCurrentTheme();

    swatchBtns.forEach(btn => {
        if (btn.dataset.theme === currentTheme) {
            btn.classList.add("active");
        } else {
            btn.classList.remove("active");
        }

        btn.addEventListener("click", () => {
            const theme = btn.dataset.theme;
            document.documentElement.setAttribute("data-theme", theme);
            localStorage.setItem("agri_theme", theme);

            swatchBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            // Re-render charts instantly with updated theme colors from memory
            loadDashboardData(false);
            loadFeatureImportances(false);
        });
    });

    const quickToggle = document.getElementById("quick-theme-toggle");
    if (quickToggle) {
        quickToggle.addEventListener("click", () => {
            const cur = getCurrentTheme();
            const next = cur === "nordic-frost" ? "obsidian-emerald" : "nordic-frost";
            document.documentElement.setAttribute("data-theme", next);
            localStorage.setItem("agri_theme", next);
            swatchBtns.forEach(b => b.classList.toggle("active", b.dataset.theme === next));
            loadDashboardData(false);
            loadFeatureImportances(false);
        });
    }
}

/* --------------------------------------------------------------------------
   Navigation Tabs
   -------------------------------------------------------------------------- */
function setupNavigation() {
    const navItems = document.querySelectorAll("#nav-menu li");
    const views = document.querySelectorAll(".view-section");

    navItems.forEach(item => {
        item.addEventListener("click", (e) => {
            e.preventDefault();
            const target = item.dataset.target;

            navItems.forEach(n => n.classList.remove("active"));
            views.forEach(v => v.classList.remove("active"));

            item.classList.add("active");
            const targetView = document.getElementById(target);
            if (targetView) targetView.classList.add("active");
        });
    });

    // Launchpad jump cards
    document.querySelectorAll(".launchpad-card").forEach(card => {
        card.addEventListener("click", () => {
            const jumpTarget = card.dataset.jump;
            if (!jumpTarget) return;
            const targetNav = document.querySelector(`#nav-menu li[data-target="${jumpTarget}"]`);
            if (targetNav) {
                targetNav.click();
            }
        });
    });
}

/* --------------------------------------------------------------------------
   Dashboard Overview & Chart.js
   -------------------------------------------------------------------------- */
async function loadDashboardData(force = false) {
    try {
        let data = cachedDashboardData;
        if (!data || force) {
            const res = await fetch("/api/data");
            if (!res.ok) return;
            data = await res.json();
            cachedDashboardData = data;
        }

        // Populate KPIs
        const kpis = data.kpis;
        document.getElementById("kpi-total-records").innerText = kpis.total_records ? Number(kpis.total_records).toLocaleString() : "0";
        document.getElementById("kpi-avg-yield").innerText = `${kpis.avg_yield || 0} t/ha`;
        document.getElementById("kpi-total-area").innerText = `${(kpis.total_area || 0).toLocaleString()} ha`;
        document.getElementById("kpi-avg-rainfall").innerText = `${kpis.avg_rainfall || 0} mm`;
        const quality = kpis.total_records > 0 ? (kpis.cleaned_records / kpis.total_records * 100).toFixed(1) : 100;
        document.getElementById("kpi-data-quality").innerText = `${quality}%`;

        const pal = getPalette();

        // 1. Yield Histogram Chart
        if (data.charts.yield_hist && data.charts.yield_hist.labels) {
            const ctx1 = document.getElementById("yieldHistChart").getContext("2d");
            if (yieldChartInstance) yieldChartInstance.destroy();
            yieldChartInstance = new Chart(ctx1, {
                type: "bar",
                data: {
                    labels: data.charts.yield_hist.labels,
                    datasets: [{
                        label: "Observations",
                        data: data.charts.yield_hist.data,
                        backgroundColor: pal.barPrimary,
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { ticks: { color: pal.textColor }, grid: { color: pal.gridColor } },
                        y: { ticks: { color: pal.textColor }, grid: { color: pal.gridColor } }
                    }
                }
            });
        }

        // 2. Seasonal Area Doughnut Chart
        if (data.charts.season_data && data.charts.season_data.labels) {
            const ctx2 = document.getElementById("seasonPieChart").getContext("2d");
            if (seasonPieInstance) seasonPieInstance.destroy();
            seasonPieInstance = new Chart(ctx2, {
                type: "doughnut",
                data: {
                    labels: data.charts.season_data.labels,
                    datasets: [{
                        data: data.charts.season_data.data,
                        backgroundColor: ['#10b981', '#38bdf8', '#8b5cf6', '#f59e0b', '#f43f5e', '#06b6d4'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        legend: { position: 'bottom', labels: { color: pal.textColor, boxWidth: 12 } }
                    }
                }
            });
        }

        // 3. Top Crops Bar Chart
        if (data.charts.top_crops && data.charts.top_crops.labels) {
            const ctx3 = document.getElementById("topCropsChart").getContext("2d");
            if (topCropsInstance) topCropsInstance.destroy();
            topCropsInstance = new Chart(ctx3, {
                type: "bar",
                data: {
                    labels: data.charts.top_crops.labels,
                    datasets: [{
                        label: "Mean Yield (t/ha)",
                        data: data.charts.top_crops.data,
                        backgroundColor: pal.barSecondary,
                        borderRadius: 6
                    }]
                },
                options: {
                    responsive: true,
                    indexAxis: 'y',
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { ticks: { color: pal.textColor }, grid: { color: pal.gridColor } },
                        y: { ticks: { color: pal.textColor }, grid: { color: pal.gridColor } }
                    }
                }
            });
        }

        // 4. Top States Bar Chart
        if (data.charts.top_states && data.charts.top_states.labels) {
            const canvas4 = document.getElementById("topStatesChart");
            if (canvas4) {
                const ctx4 = canvas4.getContext("2d");
                if (topStatesInstance) topStatesInstance.destroy();
                topStatesInstance = new Chart(ctx4, {
                    type: "bar",
                    data: {
                        labels: data.charts.top_states.labels,
                        datasets: [{
                            label: "Mean Yield (t/ha)",
                            data: data.charts.top_states.data,
                            backgroundColor: pal.barPrimary,
                            borderRadius: 6
                        }]
                    },
                    options: {
                        responsive: true,
                        indexAxis: 'y',
                        plugins: { legend: { display: false } },
                        scales: {
                            x: { ticks: { color: pal.textColor }, grid: { color: pal.gridColor } },
                            y: { ticks: { color: pal.textColor }, grid: { color: pal.gridColor } }
                        }
                    }
                });
            }
        }

    } catch (err) {
        console.error("Failed to load dashboard data:", err);
    }
}

/* --------------------------------------------------------------------------
   Metadata & Predictor Form
   -------------------------------------------------------------------------- */
async function loadMetadata() {
    try {
        const res = await fetch("/api/metadata");
        if (!res.ok) return;
        const meta = await res.json();

        const cropSelect = document.getElementById("input-crop");
        const stateSelect = document.getElementById("input-state");
        const seasonSelect = document.getElementById("input-season");

        const recStateSelect = document.getElementById("rec-state");
        const recSeasonSelect = document.getElementById("rec-season");

        const optCropSelect = document.getElementById("opt-crop");
        const optStateSelect = document.getElementById("opt-state");
        const optSeasonSelect = document.getElementById("opt-season");

        meta.crops.forEach(c => {
            if (cropSelect) cropSelect.add(new Option(c, c));
            if (optCropSelect) optCropSelect.add(new Option(c, c));
        });
        meta.states.forEach(s => {
            if (stateSelect) stateSelect.add(new Option(s, s));
            if (recStateSelect) recStateSelect.add(new Option(s, s));
            if (optStateSelect) optStateSelect.add(new Option(s, s));
        });
        meta.seasons.forEach(s => {
            if (seasonSelect) seasonSelect.add(new Option(s, s));
            if (recSeasonSelect) recSeasonSelect.add(new Option(s, s));
            if (optSeasonSelect) optSeasonSelect.add(new Option(s, s));
        });

        // Defaults
        if (meta.crops.includes("Rice") && cropSelect) cropSelect.value = "Rice";
        if (meta.crops.includes("Wheat") && optCropSelect) optCropSelect.value = "Wheat";
        if (meta.states.includes("Punjab")) {
            if (stateSelect) stateSelect.value = "Punjab";
            if (recStateSelect) recStateSelect.value = "Punjab";
            if (optStateSelect) optStateSelect.value = "Punjab";
        }
        if (meta.seasons.includes("Kharif")) {
            if (seasonSelect) seasonSelect.value = "Kharif";
            if (recSeasonSelect) recSeasonSelect.value = "Kharif";
        }
        if (meta.seasons.includes("Rabi") && optSeasonSelect) {
            optSeasonSelect.value = "Rabi";
        }

    } catch (e) {
        console.error("Failed to load metadata:", e);
    }
}

function setupPredictorForm() {
    const form = document.getElementById("prediction-form");
    if (!form) return;

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const btn = document.getElementById("predict-btn");
        btn.innerHTML = `<i class="ph ph-spinner ph-spin"></i> Analyzing...`;
        btn.disabled = true;

        const payload = {
            crop: document.getElementById("input-crop").value,
            state: document.getElementById("input-state").value,
            season: document.getElementById("input-season").value,
            crop_year: parseInt(document.getElementById("input-year").value),
            area: parseFloat(document.getElementById("input-area").value),
            annual_rainfall: parseFloat(document.getElementById("input-rainfall").value),
            fertilizer: parseFloat(document.getElementById("input-fertilizer").value),
            pesticide: parseFloat(document.getElementById("input-pesticide").value)
        };

        try {
            const res = await fetch("/api/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            const result = await res.json();

            // Populate Output
            document.getElementById("res-yield").innerText = `${result.predicted_yield_t_ha} t/ha`;
            document.getElementById("res-total-prod").innerText = `Estimated Total Production: ${Number(result.estimated_production_tonnes).toLocaleString()} tonnes`;
            document.getElementById("res-persona-name").innerText = result.matched_persona;
            document.getElementById("res-persona-name").style.color = result.zone_color || "var(--accent-primary)";

            // Tips list
            const tipsList = document.getElementById("res-tips-list");
            tipsList.innerHTML = "";
            (result.actionable_tips || []).forEach(tip => {
                const li = document.createElement("li");
                li.innerText = tip;
                tipsList.appendChild(li);
            });

            // Sensitivity Chart
            if (result.sensitivity && result.sensitivity.rainfall_steps) {
                const ctx = document.getElementById("sensitivityChart").getContext("2d");
                if (sensitivityChartInstance) sensitivityChartInstance.destroy();
                const pal = getPalette();
                sensitivityChartInstance = new Chart(ctx, {
                    type: "line",
                    data: {
                        labels: result.sensitivity.rainfall_steps.map(r => `${r}mm`),
                        datasets: [{
                            label: "Projected Yield (t/ha)",
                            data: result.sensitivity.yield_projections,
                            borderColor: pal.barPrimary,
                            backgroundColor: 'rgba(16, 185, 129, 0.15)',
                            fill: true,
                            tension: 0.3
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: { legend: { display: false } },
                        scales: {
                            x: { ticks: { color: pal.textColor }, grid: { color: pal.gridColor } },
                            y: { ticks: { color: pal.textColor }, grid: { color: pal.gridColor } }
                        }
                    }
                });
            }

        } catch (err) {
            console.error("Prediction failed:", err);
        } finally {
            btn.innerHTML = `<i class="ph-bold ph-lightning"></i> Calculate Expected Yield`;
            btn.disabled = false;
        }
    });
}

/* --------------------------------------------------------------------------
   Records Database Table
   -------------------------------------------------------------------------- */
async function loadRecords(page = 1) {
    try {
        const tbody = document.getElementById("records-table-body");
        tbody.innerHTML = `<tr><td colspan="8" style="text-align: center;">Loading page ${page}...</td></tr>`;

        const query = encodeURIComponent(currentSearch);
        const res = await fetch(`/api/records?page=${page}&limit=20&search=${query}`);
        if (!res.ok) return;
        const data = await res.json();

        currentPage = data.page;
        totalPages = data.total_pages;

        document.getElementById("page-indicator").innerText = `Page ${currentPage} of ${totalPages || 1}`;
        document.getElementById("prev-page-btn").disabled = currentPage <= 1;
        document.getElementById("next-page-btn").disabled = currentPage >= totalPages;
        document.getElementById("table-record-count").innerText = `Showing ${data.records.length} of ${data.total.toLocaleString()} records`;

        tbody.innerHTML = "";
        if (data.records.length === 0) {
            tbody.innerHTML = `<tr><td colspan="8" style="text-align: center;">No matching agricultural records found.</td></tr>`;
            return;
        }

        data.records.forEach(r => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong>${r.Crop || 'N/A'}</strong></td>
                <td>${r.State || 'N/A'}</td>
                <td><span style="background:var(--btn-bg); padding:3px 8px; border-radius:6px; font-size:0.8rem;">${r.Season || 'N/A'}</span></td>
                <td>${r.Crop_Year || 'N/A'}</td>
                <td>${r.Area ? Number(r.Area).toLocaleString() : '0'}</td>
                <td>${r.Annual_Rainfall ? Number(r.Annual_Rainfall).toFixed(1) : '0'}</td>
                <td>${r.Fertilizer ? Number(r.Fertilizer).toLocaleString() : '0'}</td>
                <td><strong style="color:var(--accent-primary);">${r.Yield ? Number(r.Yield).toFixed(2) : '0.00'}</strong></td>
            `;
            tbody.appendChild(tr);
        });

    } catch (e) {
        console.error("Failed to load records:", e);
    }
}

// Search & Pagination Event Listeners
const searchInput = document.getElementById("record-search");
if (searchInput) {
    let timeout = null;
    searchInput.addEventListener("input", (e) => {
        clearTimeout(timeout);
        timeout = setTimeout(() => {
            currentSearch = e.target.value;
            loadRecords(1);
        }, 300);
    });
}

const prevBtn = document.getElementById("prev-page-btn");
const nextBtn = document.getElementById("next-page-btn");
if (prevBtn) prevBtn.addEventListener("click", () => { if (currentPage > 1) loadRecords(currentPage - 1); });
if (nextBtn) nextBtn.addEventListener("click", () => { if (currentPage < totalPages) loadRecords(currentPage + 1); });

/* --------------------------------------------------------------------------
   Visual Reports Gallery & Modal
   -------------------------------------------------------------------------- */
let cachedGalleryPlots = [];

function categorizePlot(filename) {
    const fn = filename.toLowerCase();
    if (fn.includes("pca_") || fn.startsWith("pca_") || fn.includes("pca_2d") || fn.includes("pca_scree")) {
        return "pca";
    }
    if (fn.includes("anomaly") || fn.includes("outlier") || fn.includes("anomalous")) {
        return "anomaly";
    }
    if (fn.startsWith("boxplot_") || fn.startsWith("01_distribution_") || fn.startsWith("02_boxplot_") ||
        fn.includes("yield_distribution") || fn.includes("correlation_heatmap") ||
        fn.includes("scatter_rainfall") || fn.includes("season_vs_yield") || fn.includes("14_correlation")) {
        return "eda";
    }
    if (fn.includes("dbscan") || fn.includes("k_distance") || fn.includes("dendrogram") ||
        fn.includes("kmeans") || fn.includes("cluster") || fn.includes("silhouette")) {
        return "clustering";
    }
    if (fn.includes("actual_vs_predicted") || fn.includes("shap") || fn.includes("permutation") ||
        fn.includes("feature_importance") || fn.includes("co2_coefficient")) {
        return "models";
    }
    if (fn.startsWith("eda_") || fn.includes("comparison") || fn.includes("trend")) {
        return "trends";
    }
    return "other";
}

async function loadReportsGallery() {
    try {
        const res = await fetch("/api/plots-list");
        if (!res.ok) return;
        cachedGalleryPlots = await res.json();

        // Update button counts
        const countMap = { all: cachedGalleryPlots.length, pca: 0, anomaly: 0, clustering: 0, models: 0, eda: 0, trends: 0 };
        cachedGalleryPlots.forEach(f => {
            const cat = categorizePlot(f);
            if (countMap[cat] !== undefined) countMap[cat]++;
        });

        document.querySelectorAll(".gallery-filter-btn").forEach(btn => {
            const filter = btn.dataset.filter;
            const count = countMap[filter] !== undefined ? countMap[filter] : 0;
            const label = btn.textContent.replace(/\s*\(\d+\)$/, '');
            btn.textContent = `${label} (${count})`;

            btn.onclick = () => {
                document.querySelectorAll(".gallery-filter-btn").forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                renderGalleryGrid(filter);
            };
        });

        renderGalleryGrid("all");

    } catch (e) {
        console.error("Failed to load reports gallery:", e);
    }
}

function renderGalleryGrid(filter) {
    const grid = document.getElementById("gallery-grid");
    if (!grid) return;
    grid.innerHTML = "";

    const filtered = cachedGalleryPlots.filter(f => {
        if (filter === "all") return true;
        return categorizePlot(f) === filter;
    });

    if (filtered.length === 0) {
        grid.innerHTML = `<p style="color:var(--text-muted); grid-column:1/-1;">No visual plots found in this category.</p>`;
        return;
    }

    filtered.forEach(filename => {
        const card = document.createElement("div");
        card.className = "gallery-card";
        const cleanTitle = filename.replace(/\.png$/i, '').replace(/_/g, ' ');
        const cat = categorizePlot(filename);
        let tag = "Diagnostic Plot";
        if (cat === "pca") tag = "PCA Latent Space & Vectors";
        else if (cat === "anomaly") tag = "Agricultural Outlier Diagnostics";
        else if (cat === "eda") tag = "Quick EDA Profile";
        else if (cat === "clustering") tag = "Clustering & DBSCAN";
        else if (cat === "models") tag = "ML Model Evaluation";
        else if (cat === "trends") tag = "Regional Trend Analysis";

        card.innerHTML = `
            <img src="/api/plots/${filename}" alt="${cleanTitle}" loading="lazy">
            <div class="gallery-card-body">
                <h4>${cleanTitle}</h4>
                <span>${tag}</span>
            </div>
        `;
        card.addEventListener("click", () => openModal(`/api/plots/${filename}`));
        grid.appendChild(card);
    });
}

/* --------------------------------------------------------------------------
   Integrated Report Hub & Diagnostics
   -------------------------------------------------------------------------- */
const REPORT_METADATA = {
    pca: {
        title: "Principal Component Analysis (PCA) Technical Report",
        meta: "Standardized decomposition of 15,751 agricultural records • Session 26"
    },
    anomaly: {
        title: "Agricultural Anomaly Detection & Diagnostics Report",
        meta: "4-Algorithm Ensemble (Isolation Forest, LOF, Elliptic Envelope, OC-SVM) • Session 27"
    },
    dbscan: {
        title: "DBSCAN Agro-Ecological Density Clustering Report",
        meta: "Density partitioning and noise farm identification • Session 25"
    },
    hierarchical: {
        title: "Hierarchical Agglomerative Clustering (Dendrograms) Report",
        meta: "Ward linkage criteria with Cophenetic correlation & K=3 cut heights"
    }
};

function setupReportHub() {
    // 1. Subtab Switching
    const subnavBtns = document.querySelectorAll(".report-subnav-btn");
    const subtabPanes = document.querySelectorAll(".report-subtab-pane");

    subnavBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetId = btn.dataset.subtab;
            subnavBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");

            subtabPanes.forEach(pane => {
                if (pane.id === targetId) {
                    pane.style.display = "block";
                } else {
                    pane.style.display = "none";
                }
            });

            if (targetId === "subtab-technical") {
                const activeReport = document.querySelector(".report-pill-btn.active")?.dataset.report || "pca";
                loadTechnicalReport(activeReport);
            } else if (targetId === "subtab-anomalies") {
                loadFlaggedAnomaliesTable();
            }
        });
    });

    // 2. Report Selector Pills
    const pillBtns = document.querySelectorAll(".report-pill-btn");
    pillBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            pillBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            loadTechnicalReport(btn.dataset.report);
        });
    });

    // 3. Copy Report Text
    const copyBtn = document.getElementById("copy-report-btn");
    if (copyBtn) {
        copyBtn.addEventListener("click", () => {
            const pre = document.getElementById("technical-report-content");
            if (pre) {
                navigator.clipboard.writeText(pre.innerText).then(() => {
                    const originalText = copyBtn.innerHTML;
                    copyBtn.innerHTML = '<i class="ph-bold ph-check"></i> Copied!';
                    setTimeout(() => { copyBtn.innerHTML = originalText; }, 2000);
                });
            }
        });
    }

    // Pre-load default technical report
    loadTechnicalReport("pca");
}

async function loadTechnicalReport(reportId) {
    const pre = document.getElementById("technical-report-content");
    const titleEl = document.getElementById("current-report-title");
    const metaEl = document.getElementById("current-report-meta");
    if (!pre) return;

    if (REPORT_METADATA[reportId]) {
        if (titleEl) titleEl.textContent = REPORT_METADATA[reportId].title;
        if (metaEl) metaEl.textContent = REPORT_METADATA[reportId].meta;
    }

    pre.textContent = `Loading ${reportId} report...`;

    try {
        const res = await fetch(`/api/reports/${reportId}`);
        if (!res.ok) {
            pre.textContent = `Report '${reportId}' is currently being generated or not found.`;
            return;
        }
        const data = await res.json();
        pre.textContent = data.content || "No report text available.";
    } catch (e) {
        pre.textContent = `Failed to fetch report: ${e.message}`;
    }
}

async function loadFlaggedAnomaliesTable() {
    const tbody = document.getElementById("anomalies-table-body");
    if (!tbody) return;

    try {
        const res = await fetch("/api/reports/flagged-anomalies");
        if (!res.ok) return;
        const data = await res.json();
        const records = data.records || [];

        if (records.length === 0) {
            tbody.innerHTML = `<tr><td colspan="10" style="text-align: center; color: var(--text-muted);">No flagged anomalies recorded.</td></tr>`;
            return;
        }

        tbody.innerHTML = records.map(r => {
            const isHigh = (r.Anomaly_Category === "High-Confidence Anomaly" || r.Ensemble_Anomaly_Count >= 3);
            const badgeClass = isHigh ? "badge-anomaly-high" : "badge-anomaly-mod";
            const severityLabel = isHigh ? "High-Severity" : "Moderate";
            const fertHa = r.Fertilizer_per_Ha ? Number(r.Fertilizer_per_Ha).toFixed(1) : "N/A";
            const areaVal = r.Area ? Number(r.Area).toLocaleString() : "0";
            const yieldVal = r.Yield ? Number(r.Yield).toFixed(2) : "0.00";
            const rainVal = r.Annual_Rainfall ? Number(r.Annual_Rainfall).toFixed(1) : "0";

            return `
                <tr>
                    <td><strong>${r.Crop || "N/A"}</strong></td>
                    <td>${r.State || "N/A"}</td>
                    <td>${r.Season || "N/A"}</td>
                    <td>${r.Crop_Year || "N/A"}</td>
                    <td>${areaVal}</td>
                    <td>${rainVal} mm</td>
                    <td><strong>${fertHa}</strong> kg/ha</td>
                    <td>${yieldVal} t/ha</td>
                    <td><span class="${badgeClass}">${severityLabel} (${r.Ensemble_Anomaly_Count || 2}/4)</span></td>
                    <td style="font-size: 0.8rem; color: var(--text-secondary);">${r.Root_Cause || "Multivariate Outlier"}</td>
                </tr>
            `;
        }).join("");

    } catch (e) {
        tbody.innerHTML = `<tr><td colspan="10" style="text-align: center; color: var(--accent-danger);">Error loading flagged anomalies table: ${e.message}</td></tr>`;
    }
}

function setupModal() {
    const modal = document.getElementById("image-modal");
    const closeBtn = document.getElementById("close-modal-btn");
    if (!modal) return;

    closeBtn.addEventListener("click", () => modal.style.display = "none");
    modal.addEventListener("click", (e) => {
        if (e.target === modal) modal.style.display = "none";
    });
}

function openModal(src) {
    const modal = document.getElementById("image-modal");
    const img = document.getElementById("modal-img");
    if (modal && img) {
        img.src = src;
        modal.style.display = "flex";
    }
}

/* --------------------------------------------------------------------------
   Model Benchmarks & Feature Importances
   -------------------------------------------------------------------------- */
async function loadModelsComparison() {
    try {
        const tbody = document.getElementById("models-table-body");
        if (!tbody) return;
        const res = await fetch("/api/models/comparison");
        if (!res.ok) return;
        const models = await res.json();

        tbody.innerHTML = "";
        models.forEach(m => {
            const tr = document.createElement("tr");
            const stat = m.Status || (m.Model && m.Model.includes("Random Forest") ? "Champion" : "Standard");
            let badgeBg = "rgba(100, 116, 139, 0.15)";
            let badgeColor = "#94a3b8";
            if (stat === "Champion") {
                badgeBg = "rgba(16, 185, 129, 0.15)";
                badgeColor = "#10b981";
            } else if (stat === "High-Accuracy") {
                badgeBg = "rgba(56, 189, 248, 0.15)";
                badgeColor = "#38bdf8";
            } else if (stat === "Regularized") {
                badgeBg = "rgba(168, 85, 247, 0.15)";
                badgeColor = "#a855f7";
            }

            const timeStr = m.Training_Time_Sec !== undefined ? `${Number(m.Training_Time_Sec).toFixed(2)}s` : "< 1s";

            tr.innerHTML = `
                <td><strong>${m.Model}</strong></td>
                <td><strong style="color:var(--accent-primary);">${Number(m.R2_Score).toFixed(4)}</strong></td>
                <td>${Number(m.RMSE).toFixed(2)}</td>
                <td>${Number(m.MAE).toFixed(2)}</td>
                <td><span style="font-family:'JetBrains Mono',monospace; font-size:0.8rem;">${timeStr}</span></td>
                <td><span style="background:${badgeBg}; color:${badgeColor}; padding:3px 8px; border-radius:6px; font-weight:600; font-size:0.75rem;">${stat}</span></td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        console.error("Failed to load models comparison:", e);
    }
}

async function loadFeatureImportances(force = false) {
    try {
        const canvas = document.getElementById("featureImportanceChart");
        if (!canvas) return;
        let features = cachedFeatureData;
        if (!features || force) {
            const res = await fetch("/api/models/feature-importances");
            if (!res.ok) return;
            features = await res.json();
            cachedFeatureData = features;
        }

        const pal = getPalette();
        const ctx = canvas.getContext("2d");
        if (featureImportanceInstance) featureImportanceInstance.destroy();

        featureImportanceInstance = new Chart(ctx, {
            type: "bar",
            data: {
                labels: features.map(f => f.Feature),
                datasets: [{
                    label: "Relative Importance",
                    data: features.map(f => Number(f.Importance)),
                    backgroundColor: pal.barPrimary,
                    borderRadius: 6
                }]
            },
            options: {
                responsive: true,
                indexAxis: 'y',
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { color: pal.textColor }, grid: { color: pal.gridColor } },
                    y: { ticks: { color: pal.textColor }, grid: { color: pal.gridColor } }
                }
            }
        });
    } catch (e) {
        console.error("Failed to load feature importances:", e);
    }
}

/* --------------------------------------------------------------------------
   Smart Crop Recommender
   -------------------------------------------------------------------------- */
function setupCropRecommender() {
    const form = document.getElementById("crop-recommend-form");
    if (!form) return;

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const btn = document.getElementById("rec-crops-btn");
        const origText = btn.innerHTML;
        btn.innerHTML = `<i class="ph-bold ph-spinner animate-spin"></i> Analyzing Agro-Data...`;
        btn.disabled = true;

        try {
            const payload = {
                state: document.getElementById("rec-state").value,
                season: document.getElementById("rec-season").value,
                area: parseFloat(document.getElementById("rec-area").value) || 1000,
                annual_rainfall: parseFloat(document.getElementById("rec-rainfall").value) || 1100,
                fertilizer: parseFloat(document.getElementById("rec-fert").value) || 80000,
                pesticide: parseFloat(document.getElementById("rec-pest").value) || 600
            };

            const res = await fetch("/api/recommend-crops", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!res.ok) throw new Error("Failed to evaluate crops");
            const data = await res.json();

            const container = document.getElementById("rec-results-container");
            const grid = document.getElementById("rec-cards-grid");
            grid.innerHTML = "";

            if (data.top_recommendations && data.top_recommendations.length > 0) {
                data.top_recommendations.forEach(rec => {
                    const card = document.createElement("div");
                    card.className = "rec-card";
                    card.innerHTML = `
                        <div>
                            <span class="rec-badge ${rec.badge_color}">${rec.badge}</span>
                            <div class="rec-crop-name">${rec.crop}</div>
                            <div class="rec-stats">
                                <div class="rec-stat-item">
                                    <span>Predicted Yield</span>
                                    <strong>${rec.predicted_yield_t_ha} t/ha</strong>
                                </div>
                                <div class="rec-stat-item">
                                    <span>Total Harvest</span>
                                    <strong>${Number(rec.estimated_production_tonnes).toLocaleString()} t</strong>
                                </div>
                                <div class="rec-stat-item" style="grid-column: span 2;">
                                    <span>Water Efficiency (kg / mm rain)</span>
                                    <strong>${rec.water_efficiency_kg_per_mm} kg/mm</strong>
                                </div>
                            </div>
                        </div>
                        <button class="btn-select-crop" data-crop="${rec.crop}">
                            <i class="ph-bold ph-arrow-down"></i> Optimize Dosage for ${rec.crop}
                        </button>
                    `;

                    card.querySelector(".btn-select-crop").addEventListener("click", () => {
                        const optCrop = document.getElementById("opt-crop");
                        const optState = document.getElementById("opt-state");
                        const optSeason = document.getElementById("opt-season");
                        const optArea = document.getElementById("opt-area");
                        const optRain = document.getElementById("opt-rainfall");
                        const optFert = document.getElementById("opt-fertilizer");

                        if (optCrop) optCrop.value = rec.crop;
                        if (optState) optState.value = payload.state;
                        if (optSeason) optSeason.value = payload.season;
                        if (optArea) optArea.value = payload.area;
                        if (optRain) optRain.value = payload.annual_rainfall;
                        if (optFert) optFert.value = payload.fertilizer;

                        document.getElementById("optimizer-form").scrollIntoView({ behavior: 'smooth' });
                        document.getElementById("opt-run-btn").click();
                    });

                    grid.appendChild(card);
                });
                container.style.display = "block";
            }
        } catch (err) {
            console.error("Crop recommendation error:", err);
            alert("Could not load crop recommendations. Please check server connection.");
        } finally {
            btn.innerHTML = origText;
            btn.disabled = false;
        }
    });
}

/* --------------------------------------------------------------------------
   Fertilizer Dosage & Input Sweet-Spot Optimizer
   -------------------------------------------------------------------------- */
let optimizerCurveChartInstance = null;

function setupDosageOptimizer() {
    const form = document.getElementById("optimizer-form");
    if (!form) return;

    form.addEventListener("submit", async (e) => {
        e.preventDefault();
        const btn = document.getElementById("opt-run-btn");
        const origText = btn.innerHTML;
        btn.innerHTML = `<i class="ph-bold ph-spinner animate-spin"></i> Calculating Sweet Spot...`;
        btn.disabled = true;

        try {
            const payload = {
                crop: document.getElementById("opt-crop").value,
                state: document.getElementById("opt-state").value,
                season: document.getElementById("opt-season").value,
                area: parseFloat(document.getElementById("opt-area").value) || 1000,
                annual_rainfall: parseFloat(document.getElementById("opt-rainfall").value) || 1100,
                fertilizer: parseFloat(document.getElementById("opt-fertilizer").value) || 90000,
                pesticide: 650.0
            };

            const res = await fetch("/api/optimize-inputs", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });

            if (!res.ok) throw new Error("Optimizer computation failed");
            const data = await res.json();

            const container = document.getElementById("optimizer-results-container");
            const summaryText = document.getElementById("opt-summary-text");
            summaryText.textContent = data.recommendation_summary;
            container.style.display = "block";

            // Render Curve Chart
            const canvas = document.getElementById("optimizerCurveChart");
            if (canvas && data.response_curve) {
                const pal = getPalette();
                const ctx = canvas.getContext("2d");
                if (optimizerCurveChartInstance) optimizerCurveChartInstance.destroy();

                const labels = data.response_curve.map(pt => `${pt.pct_label} (${(pt.fertilizer_kg/1000).toFixed(0)}k kg)`);
                const yields = data.response_curve.map(pt => pt.predicted_yield_t_ha);
                const pointBg = data.response_curve.map(pt => pt.is_recommended ? '#10b981' : (pt.is_baseline ? '#f59e0b' : pal.barSecondary));
                const pointRadius = data.response_curve.map(pt => (pt.is_recommended || pt.is_baseline) ? 8 : 4);

                optimizerCurveChartInstance = new Chart(ctx, {
                    type: "line",
                    data: {
                        labels: labels,
                        datasets: [{
                            label: "Forecast Yield (t/ha)",
                            data: yields,
                            borderColor: pal.barPrimary,
                            backgroundColor: "rgba(16, 185, 129, 0.08)",
                            fill: true,
                            tension: 0.35,
                            pointBackgroundColor: pointBg,
                            pointBorderColor: "#fff",
                            pointBorderWidth: 2,
                            pointRadius: pointRadius
                        }]
                    },
                    options: {
                        responsive: true,
                        plugins: {
                            legend: { display: false },
                            tooltip: {
                                callbacks: {
                                    afterLabel: (ctx) => {
                                        const pt = data.response_curve[ctx.dataIndex];
                                        if (pt.is_recommended) return "★ Recommended Agronomic Sweet Spot";
                                        if (pt.is_baseline) return "● Current Baseline Dosage";
                                        return "";
                                    }
                                }
                            }
                        },
                        scales: {
                            x: { ticks: { color: pal.textColor }, grid: { color: pal.gridColor } },
                            y: { ticks: { color: pal.textColor }, grid: { color: pal.gridColor } }
                        }
                    }
                });
            }
        } catch (err) {
            console.error("Optimizer error:", err);
            alert("Could not calculate input sweet spot.");
        } finally {
            btn.innerHTML = origText;
            btn.disabled = false;
        }
    });

    const printBtn = document.getElementById("print-advisory-btn");
    if (printBtn) {
        printBtn.addEventListener("click", () => {
            window.print();
        });
    }
}

/* --------------------------------------------------------------------------
   Batch Multi-Farm Simulation & CSV Export
   -------------------------------------------------------------------------- */
let currentBatchRecords = [];
let currentBatchResults = [];

function setupBatchSimulation() {
    const presets = {
        smallholder: [
            { crop: "Rice", state: "Punjab", season: "Kharif", area: 3.5, annual_rainfall: 1150, fertilizer: 420, pesticide: 3.5 },
            { crop: "Wheat", state: "Punjab", season: "Rabi", area: 3.5, annual_rainfall: 180, fertilizer: 450, pesticide: 3.2 },
            { crop: "Maize", state: "Uttar Pradesh", season: "Kharif", area: 2.0, annual_rainfall: 950, fertilizer: 280, pesticide: 2.0 },
            { crop: "Pulses", state: "Madhya Pradesh", season: "Rabi", area: 4.0, annual_rainfall: 800, fertilizer: 320, pesticide: 2.5 }
        ],
        commercial: [
            { crop: "Wheat", state: "Punjab", season: "Rabi", area: 1200, annual_rainfall: 950, fertilizer: 95000, pesticide: 800 },
            { crop: "Cotton(lint)", state: "Gujarat", season: "Kharif", area: 850, annual_rainfall: 820, fertilizer: 68000, pesticide: 620 },
            { crop: "Rice", state: "Andhra Pradesh", season: "Kharif", area: 1500, annual_rainfall: 1250, fertilizer: 120000, pesticide: 950 },
            { crop: "Sugarcane", state: "Maharashtra", season: "Whole Year", area: 600, annual_rainfall: 1100, fertilizer: 75000, pesticide: 540 }
        ],
        plantation: [
            { crop: "Tea", state: "Assam", season: "Whole Year", area: 450, annual_rainfall: 2600, fertilizer: 55000, pesticide: 420 },
            { crop: "Coffee", state: "Karnataka", season: "Whole Year", area: 380, annual_rainfall: 2200, fertilizer: 42000, pesticide: 310 },
            { crop: "Coconut", state: "Kerala", season: "Whole Year", area: 520, annual_rainfall: 2800, fertilizer: 60000, pesticide: 450 },
            { crop: "Rubber", state: "Kerala", season: "Whole Year", area: 300, annual_rainfall: 2900, fertilizer: 38000, pesticide: 290 }
        ],
        arid: [
            { crop: "Bajra", state: "Rajasthan", season: "Kharif", area: 350, annual_rainfall: 380, fertilizer: 18000, pesticide: 120 },
            { crop: "Jowar", state: "Maharashtra", season: "Kharif", area: 420, annual_rainfall: 520, fertilizer: 22000, pesticide: 150 },
            { crop: "Gram", state: "Rajasthan", season: "Rabi", area: 280, annual_rainfall: 320, fertilizer: 15000, pesticide: 110 },
            { crop: "Groundnut", state: "Gujarat", season: "Kharif", area: 500, annual_rainfall: 580, fertilizer: 30000, pesticide: 210 }
        ]
    };

    function loadPreset(name) {
        currentBatchRecords = presets[name] || presets.smallholder;
        const tbody = document.getElementById("batch-table-body");
        tbody.innerHTML = "";
        currentBatchRecords.forEach((r, idx) => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong>#${idx + 1}</strong></td>
                <td>${r.crop}</td>
                <td>${r.state}</td>
                <td>${r.season}</td>
                <td>${r.area}</td>
                <td>${r.annual_rainfall}</td>
                <td>${r.fertilizer}</td>
                <td colspan="4" style="color: var(--text-muted); font-style: italic;">Pending execution...</td>
            `;
            tbody.appendChild(tr);
        });
        document.getElementById("batch-kpi-grid").style.display = "none";
    }

    const pSmall = document.getElementById("preset-smallholder");
    const pComm = document.getElementById("preset-commercial");
    const pPlant = document.getElementById("preset-plantation");
    const pArid = document.getElementById("preset-arid");

    if (pSmall) pSmall.addEventListener("click", () => loadPreset("smallholder"));
    if (pComm) pComm.addEventListener("click", () => loadPreset("commercial"));
    if (pPlant) pPlant.addEventListener("click", () => loadPreset("plantation"));
    if (pArid) pArid.addEventListener("click", () => loadPreset("arid"));

    loadPreset("smallholder");

    const runBtn = document.getElementById("run-batch-sim-btn");
    if (runBtn) {
        runBtn.addEventListener("click", async () => {
            if (!currentBatchRecords || currentBatchRecords.length === 0) {
                alert("Please select a scenario preset first.");
                return;
            }

            const origText = runBtn.innerHTML;
            runBtn.innerHTML = `<i class="ph-bold ph-spinner animate-spin"></i> Processing ${currentBatchRecords.length} Farms...`;
            runBtn.disabled = true;

            try {
                const res = await fetch("/api/batch-predict", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ records: currentBatchRecords })
                });

                if (!res.ok) throw new Error("Batch prediction failed");
                const data = await res.json();
                currentBatchResults = data.results || [];

                if (data.kpis) {
                    document.getElementById("batch-kpi-area").textContent = `${Number(data.kpis.total_area_ha).toLocaleString()} ha`;
                    document.getElementById("batch-kpi-prod").textContent = `${Number(data.kpis.total_production_tonnes).toLocaleString()} t`;
                    document.getElementById("batch-kpi-yield").textContent = `${Number(data.kpis.average_yield_t_ha).toFixed(2)} t/ha`;
                    document.getElementById("batch-kpi-count").textContent = data.count;
                    document.getElementById("batch-kpi-grid").style.display = "grid";
                }

                const tbody = document.getElementById("batch-table-body");
                tbody.innerHTML = "";
                currentBatchResults.forEach(r => {
                    const tr = document.createElement("tr");
                    tr.innerHTML = `
                        <td><strong>#${r.id}</strong></td>
                        <td><strong>${r.crop}</strong></td>
                        <td>${r.state}</td>
                        <td>${r.season}</td>
                        <td>${r.area}</td>
                        <td>${r.rainfall}</td>
                        <td>${r.fertilizer}</td>
                        <td><strong style="color:var(--accent-primary);">${r.predicted_yield_t_ha} t/ha</strong></td>
                        <td><span style="font-family:'JetBrains Mono',monospace; font-size:0.8rem; color:var(--text-muted);">${r.lower_ci} - ${r.upper_ci}</span></td>
                        <td><strong>${Number(r.total_production_tonnes).toLocaleString()} t</strong></td>
                        <td><span class="rec-badge badge-sky">${r.persona}</span></td>
                    `;
                    tbody.appendChild(tr);
                });

            } catch (err) {
                console.error("Batch simulation error:", err);
                alert("Batch prediction simulation failed.");
            } finally {
                runBtn.innerHTML = origText;
                runBtn.disabled = false;
            }
        });
    }

    const exportBtn = document.getElementById("export-batch-csv-btn");
    if (exportBtn) {
        exportBtn.addEventListener("click", () => {
            if (!currentBatchResults || currentBatchResults.length === 0) {
                alert("Please run the batch simulation first before exporting.");
                return;
            }

            const headers = ["Plot_ID", "Crop", "State", "Season", "Area_ha", "Rainfall_mm", "Fertilizer_kg", "Predicted_Yield_t_ha", "Lower_CI_t_ha", "Upper_CI_t_ha", "Total_Harvest_tonnes", "Agro_Zone"];
            const rows = currentBatchResults.map(r => [
                r.id, `"${r.crop}"`, `"${r.state}"`, `"${r.season}"`, r.area, r.rainfall, r.fertilizer, r.predicted_yield_t_ha, r.lower_ci, r.upper_ci, r.total_production_tonnes, `"${r.persona}"`
            ]);

            const csvContent = "data:text/csv;charset=utf-8," + [headers.join(","), ...rows.map(e => e.join(","))].join("\n");
            const encodedUri = encodeURI(csvContent);
            const link = document.createElement("a");
            link.setAttribute("href", encodedUri);
            link.setAttribute("download", `AgriYield_Batch_Simulation_${Date.now()}.csv`);
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        });
    }
}
