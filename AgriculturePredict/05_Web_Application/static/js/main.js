let currentPage = 1;
let totalPages = 1;
let currentSearch = "";
let yieldChartInstance = null;
let seasonPieInstance = null;
let topCropsInstance = null;
let sensitivityChartInstance = null;

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
    setupPredictorForm();
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

            // Re-render charts with updated theme colors
            loadDashboardData();
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
            loadDashboardData();
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
}

/* --------------------------------------------------------------------------
   Dashboard Overview & Chart.js
   -------------------------------------------------------------------------- */
async function loadDashboardData() {
    try {
        const res = await fetch("/api/data");
        if (!res.ok) return;
        const data = await res.json();

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

        meta.crops.forEach(c => cropSelect.add(new Option(c, c)));
        meta.states.forEach(s => stateSelect.add(new Option(s, s)));
        meta.seasons.forEach(s => seasonSelect.add(new Option(s, s)));

        // Defaults
        if (meta.crops.includes("Rice")) cropSelect.value = "Rice";
        if (meta.states.includes("Punjab")) stateSelect.value = "Punjab";
        if (meta.seasons.includes("Kharif")) seasonSelect.value = "Kharif";

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
async function loadReportsGallery() {
    try {
        const res = await fetch("/api/plots-list");
        if (!res.ok) return;
        const plots = await res.json();

        const grid = document.getElementById("gallery-grid");
        grid.innerHTML = "";

        if (plots.length === 0) {
            grid.innerHTML = `<p style="color:var(--text-muted);">No generated report plots found. Run the EDA pipeline to create visualizations.</p>`;
            return;
        }

        plots.forEach(filename => {
            const card = document.createElement("div");
            card.className = "gallery-card";
            const cleanTitle = filename.replace(/\.png$/i, '').replace(/_/g, ' ');
            card.innerHTML = `
                <img src="/api/plots/${filename}" alt="${cleanTitle}" loading="lazy">
                <div class="gallery-card-body">
                    <h4>${cleanTitle}</h4>
                    <span>High-resolution ML diagnostic visualization</span>
                </div>
            `;
            card.addEventListener("click", () => openModal(`/api/plots/${filename}`));
            grid.appendChild(card);
        });

    } catch (e) {
        console.error("Failed to load reports gallery:", e);
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
