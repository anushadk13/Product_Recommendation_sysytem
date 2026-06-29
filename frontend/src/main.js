import './style.css';

// --- State ---
let presetsData = [];
let currentProducts = [];

// --- DOM Elements ---
const userSelect = document.getElementById('userSelect');
const presetSelect = document.getElementById('presetSelect');
const presetDescription = document.getElementById('presetDescription');
const productGrid = document.getElementById('productGrid');
const loadingState = document.getElementById('loadingState');
const latencyDisplay = document.getElementById('latencyDisplay');
const candidatesDisplay = document.getElementById('candidatesDisplay');

// Modal Elements
const modalOverlay = document.getElementById('detailsModal');
const closeModal = document.getElementById('closeModal');
const modalTitle = document.getElementById('modalTitle');
const modalBody = document.getElementById('modalBody');

// --- Initialization ---
async function init() {
    await fetchPresets();
    setupEventListeners();
    fetchRecommendations(); // Initial fetch
}

// --- API Calls ---
async function fetchPresets() {
    try {
        const res = await fetch('/api/weight-presets');
        const data = await res.json();
        presetsData = data.presets;
        
        // Update description for initial value
        updatePresetDescription(presetSelect.value);
    } catch (e) {
        console.error("Failed to fetch presets", e);
    }
}

async function fetchRecommendations() {
    const userId = userSelect.value;
    const preset = presetSelect.value;

    productGrid.innerHTML = '';
    loadingState.classList.remove('hidden');

    try {
        const res = await fetch('/api/recommend', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                user_id: userId,
                top_k: 12,
                candidate_pool: 200,
                weight_preset: preset,
                apply_diversity: true
            })
        });

        if (!res.ok) throw new Error("API Error");
        
        const data = await res.json();
        currentProducts = data.products;
        
        // Update Metrics
        latencyDisplay.textContent = `${data.latency_ms} ms`;
        candidatesDisplay.textContent = data.total_candidates;
        
        renderGrid(data.products);
    } catch (e) {
        console.error("Failed to fetch recommendations", e);
        productGrid.innerHTML = `<div style="text-align:center; grid-column: 1/-1; color: var(--accent-red)">Error loading recommendations. Is the backend running?</div>`;
    } finally {
        loadingState.classList.add('hidden');
    }
}

// --- UI Rendering ---
function renderGrid(products) {
    productGrid.innerHTML = '';
    
    products.forEach((product, idx) => {
        // Animation delay for staggered entrance
        const delay = idx * 50; 
        
        const card = document.createElement('div');
        card.className = 'product-card glass-panel';
        card.style.animation = `fadeInUp 0.5s ease ${delay}ms both`;
        
        // Determine image based on category
        const imgName = getCategoryImage(product.category);
        
        // Build badges
        let badgesHTML = '';
        if (product.should_show_sale_badge) {
            badgesHTML += `<div class="badge sale">SALE</div>`;
        }
        if (product.demand_spike > 0.7) {
            badgesHTML += `<div class="badge trending">Trending</div>`;
        }

        card.innerHTML = `
            <div class="product-image-container">
                <div class="badges">${badgesHTML}</div>
                <div class="rank-badge">#${product.rank}</div>
                <img src="${imgName}" alt="${product.category}" class="product-image" loading="lazy" />
            </div>
            <div class="product-info">
                <div class="product-brand">${product.brand} • ${product.category}</div>
                <div class="product-title">${product.product_id}</div>
                <div class="product-price-row">
                    <span class="current-price">$${product.current_price.toFixed(2)}</span>
                    ${product.current_price < product.base_price ? `<span class="base-price">$${product.base_price.toFixed(2)}</span>` : ''}
                </div>
                <div class="score-bar-container">
                    <div class="score-bar" style="width: ${Math.min(product.final_score * 100, 100)}%"></div>
                </div>
            </div>
        `;

        card.addEventListener('click', () => openModal(product));
        productGrid.appendChild(card);
    });
}

function getCategoryImage(category) {
    const cat = category.toLowerCase();
    const map = {
        'eyewear': '/images/category_eyewear_1782690828080.png',
        'electronics': '/images/category_electronics_1782690836767.png',
        'home': '/images/category_home_1782690845718.png',
        'fashion': '/images/category_fashion_1782690855406.png',
        'beauty': '/images/category_beauty_1782690866801.png',
        'sports': '/images/category_sports_1782690876018.png'
    };
    return map[cat] || 'https://via.placeholder.com/400x300/1e293b/f8fafc?text=Product';
}

// --- Modal ---
function openModal(product) {
    modalTitle.textContent = `${product.brand} - Score Breakdown`;
    
    // Safety check for score_breakdown
    const scores = product.score_breakdown || {
        propensity: 0, inventory: 0, margin: 0, trend: 0, demand_spike: 0
    };

    const maxScore = 0.5; // used for bar scaling roughly
    
    const rows = [
        { label: 'Purchase Propensity', val: product.propensity_score, weightScore: scores.propensity, color: 'blue' },
        { label: 'Inventory Pressure', val: product.inventory_pressure, weightScore: scores.inventory, color: 'red' },
        { label: 'Margin Score', val: product.margin_score, weightScore: scores.margin, color: 'green' },
        { label: 'Trend / Velocity', val: product.trend_score, weightScore: scores.trend, color: 'orange' },
        { label: 'Demand Spike', val: product.demand_spike, weightScore: scores.demand_spike, color: 'purple' },
    ];

    let html = `
        <div style="font-size: 18px; font-weight: 700; color: white; margin-bottom: 8px;">
            Final Score: ${product.final_score.toFixed(4)}
        </div>
    `;

    rows.forEach(r => {
        // Raw score [0,1]
        const pct = Math.min(r.val * 100, 100);
        html += `
            <div class="breakdown-row">
                <div class="breakdown-header">
                    <span class="breakdown-label">${r.label}</span>
                    <span class="breakdown-value">${r.val.toFixed(3)}</span>
                </div>
                <div class="progress-track">
                    <div class="progress-fill fill-${r.color}" style="width: ${pct}%"></div>
                </div>
                <div style="font-size: 11px; color: var(--text-muted); text-align: right;">Weighted Contribution: +${r.weightScore.toFixed(3)}</div>
            </div>
        `;
    });

    if (product.optimal_discount_pct > 0) {
        html += `
            <div class="discount-card">
                <div class="discount-title">Causal ML Optimal Discount</div>
                <div class="discount-value">${(product.optimal_discount_pct * 100).toFixed(0)}% OFF</div>
                <div style="font-size: 12px; margin-top: 4px; color: rgba(255,255,255,0.7)">
                    Maximizes conversion lift vs margin loss
                </div>
            </div>
        `;
    }

    modalBody.innerHTML = html;
    modalOverlay.classList.remove('hidden');
}

// --- Event Listeners ---
function setupEventListeners() {
    userSelect.addEventListener('change', fetchRecommendations);
    
    presetSelect.addEventListener('change', (e) => {
        updatePresetDescription(e.target.value);
        fetchRecommendations();
    });

    closeModal.addEventListener('click', () => {
        modalOverlay.classList.add('hidden');
    });

    modalOverlay.addEventListener('click', (e) => {
        if (e.target === modalOverlay) {
            modalOverlay.classList.add('hidden');
        }
    });
}

function updatePresetDescription(val) {
    const preset = presetsData.find(p => p.name === val);
    if (preset) {
        presetDescription.textContent = preset.description;
    }
}

// Add simple keyframe animation dynamically
const style = document.createElement('style');
style.innerHTML = `
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
`;
document.head.appendChild(style);

// Start
init();
