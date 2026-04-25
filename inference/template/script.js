const state = {
    activeId: null,
    fields: {},
    specs: {}
};

const elements = {
    tabs: document.querySelectorAll('.tab-btn'),
    fieldsContainer: document.getElementById('fields-container'),
    inferenceForm: document.getElementById('inference-form'),
    resultDisplay: document.getElementById('result-display'),
    infoTrigger: document.querySelector('.info-trigger'),
    modal: document.getElementById('modal-overlay'),
    modalClose: document.getElementById('modal-close'),
    specsContent: document.getElementById('specs-content'),
    modalTitle: document.getElementById('modal-title'),
    btnAutofill: document.getElementById('btn-autofill')
};

async function autofill() {
    elements.btnAutofill.disabled = true;
    const btnText = elements.btnAutofill.querySelector('.btn-text');
    const originalText = btnText.innerText;
    btnText.innerText = 'Extracting...';
    
    try {
        const res = await fetch(`/api/sample/${state.activeId}`);
        const data = await res.json();
        
        Object.entries(data).forEach(([key, val]) => {
            const input = elements.inferenceForm.querySelector(`[name="${key}"]`);
            if (input) {
                input.value = val;
                input.style.borderColor = 'var(--secondary)';
                input.style.boxShadow = '0 0 15px rgba(0, 210, 255, 0.3)';
                setTimeout(() => {
                    input.style.borderColor = '';
                    input.style.boxShadow = '';
                }, 1500);
            }
        });
    } catch (e) {
        console.error(e);
    } finally {
        elements.btnAutofill.disabled = false;
        btnText.innerText = originalText;
    }
}

elements.btnAutofill.onclick = autofill;

async function loadTab(id) {
    if (state.activeId === id) return;
    
    state.activeId = id;
    elements.tabs.forEach(btn => btn.classList.toggle('active', btn.dataset.id === id));
    
    elements.fieldsContainer.style.opacity = '0';
    elements.fieldsContainer.style.transform = 'translateY(10px)';
    
    try {
        const [fields, info] = await Promise.all([
            fetch(`/api/fields/${id}`).then(r => r.json()),
            fetch(`/api/info/${id}`).then(r => r.json())
        ]);
        
        state.fields[id] = fields;
        state.specs[id] = info;
        
        renderFields(fields);
        elements.resultDisplay.innerHTML = `
            <div class="empty-state">
                <div class="neural-loader">
                    <div class="node"></div><div class="node"></div><div class="node"></div>
                </div>
                <p>System Initialized. Awaiting Input Data.</p>
            </div>
        `;
    } catch (e) {
        elements.fieldsContainer.innerHTML = `<p class="error">System Error: ${e.message}</p>`;
    } finally {
        setTimeout(() => {
            elements.fieldsContainer.style.opacity = '1';
            elements.fieldsContainer.style.transform = 'translateY(0)';
            elements.fieldsContainer.style.transition = 'all 0.5s cubic-bezier(0.2, 0.8, 0.2, 1)';
        }, 50);
    }
}

function renderFields(fields) {
    elements.fieldsContainer.innerHTML = fields.map((f, i) => `
        <div class="field-group" style="animation: slide-reveal 0.5s ease forwards; animation-delay: ${i * 0.03}s; opacity: 0;">
            <label>${f.name}</label>
            ${f.type === 'select' ? `
                <select name="${f.name}">
                    <option value="" disabled selected>Select option...</option>
                    ${f.options.map(opt => `<option value="${opt}">${opt}</option>`).join('')}
                </select>
            ` : `
                <input type="number" name="${f.name}" placeholder="Value..." step="any">
            `}
        </div>
    `).join('');
}

elements.inferenceForm.onsubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData(elements.inferenceForm);
    const payload = { features: {} };
    let hasEmpty = false;

    formData.forEach((v, k) => {
        if (v === "") hasEmpty = true;
        payload.features[k] = v;
    });

    if (hasEmpty) {
        alert("Please complete all parameters before execution.");
        return;
    }

    const btn = document.querySelector('.btn-primary');
    const btnText = btn.querySelector('.btn-text');
    const originalText = btnText.innerText;
    
    btn.disabled = true;
    btnText.innerText = 'Calculating...';

    elements.resultDisplay.scrollIntoView({ behavior: 'smooth', block: 'center' });

    try {
        const res = await fetch(`/api/run/${state.activeId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await res.json();
        
        if (!res.ok) throw new Error(data.detail || 'Inference engine failure');
        renderResult(data);
    } catch (e) {
        elements.resultDisplay.innerHTML = `
            <div class="empty-state">
                <p style="color: var(--primary)">Execution Error: ${e.message}</p>
            </div>
        `;
    } finally {
        btn.disabled = false;
        btnText.innerText = originalText;
    }
};

function renderResult(data) {
    const scores = Object.entries(data.scores).sort((a, b) => b[1] - a[1]);
    const shap = Object.entries(data.shap || {});
    const maxShap = Math.max(...shap.map(s => Math.abs(s[1])), 0.001);

    elements.resultDisplay.innerHTML = `
        <div class="result-view">
            <div class="prediction-box">
                <span style="font-size: 0.65rem; color: var(--text-dim); letter-spacing: 4px; font-weight: 700; text-transform: uppercase;">Prediction Output</span>
                <div class="label-val">${data.result}</div>
            </div>
            
            <div class="score-chart">
                <span style="font-size: 0.65rem; color: var(--text-dim); letter-spacing: 4px; font-weight: 700; text-transform: uppercase; margin-bottom: 10px;">Probability Vectors</span>
                ${scores.map(([label, score]) => `
                    <div class="score-row">
                        <span class="score-label" title="${label}">${label}</span>
                        <div class="bar-container">
                            <div class="bar-fill" style="width: 0%"></div>
                        </div>
                        <span class="score-label" style="text-align: right; width: 50px; font-weight: 700; color: var(--text)">${(score * 100).toFixed(1)}%</span>
                    </div>
                `).join('')}
            </div>

            <div class="shap-chart">
                <span style="font-size: 0.65rem; color: var(--text-dim); letter-spacing: 4px; font-weight: 700; text-transform: uppercase; margin-bottom: 10px;">SHAP Decision Drivers</span>
                ${shap.map(([feat, val]) => `
                    <div class="shap-row">
                        <span class="score-label" style="font-size: 0.75rem; overflow: hidden; text-overflow: ellipsis;" title="${feat}">${feat}</span>
                        <div class="shap-viz">
                            <div class="shap-axis"></div>
                            <div class="shap-bar ${val >= 0 ? 'positive' : 'negative'}" style="width: 0%"></div>
                        </div>
                        <span class="score-label" style="text-align: right; font-weight: 700; color: ${val >= 0 ? '#10b981' : '#ef4444'}">${val >= 0 ? '+' : ''}${val.toFixed(2)}</span>
                    </div>
                `).join('')}
            </div>
        </div>
    `;

    setTimeout(() => {
        const fills = elements.resultDisplay.querySelectorAll('.bar-fill');
        scores.forEach((s, i) => fills[i].style.width = `${s[1] * 100}%`);
        
        const shapBars = elements.resultDisplay.querySelectorAll('.shap-bar');
        shap.forEach((s, i) => {
            const width = (Math.abs(s[1]) / maxShap) * 50; // Max 50% from center
            shapBars[i].style.width = `${width}%`;
        });
    }, 100);
}

elements.infoTrigger.onclick = () => {
    const info = state.specs[state.activeId];
    if (!info) return;
    
    elements.modalTitle.innerText = `${info.title} Architecture`;
    elements.specsContent.innerHTML = `
        <div class="spec-grid">
            <div class="spec-item"><h4>Volume</h4><p>${info.dataset.rows.toLocaleString()}</p></div>
            <div class="spec-item"><h4>Dimensions</h4><p>${info.dataset.cols}</p></div>
            <div class="spec-item"><h4>Model Type</h4><p>${info.model.type || 'Ensemble'}</p></div>
            <div class="spec-item"><h4>Target</h4><p>${info.dataset.target}</p></div>
        </div>
        <div class="metrics-table-wrapper" style="margin-bottom: 30px;">
            <table style="width: 100%; border-collapse: collapse; font-family: var(--font-mono); font-size: 0.8rem;">
                <thead style="text-align: left; color: var(--text-dim); border-bottom: 1px solid var(--border);">
                    <tr>${info.metrics.length ? Object.keys(info.metrics[0]).map(k => `<th style="padding: 10px;">${k}</th>`).join('') : ''}</tr>
                </thead>
                <tbody>
                    ${info.metrics.map(m => `
                        <tr style="border-bottom: 1px solid var(--border);">
                            ${Object.values(m).map(v => `<td style="padding: 10px; color: var(--text);">${typeof v === 'number' ? v.toFixed(4) : v}</td>`).join('')}
                        </tr>
                    `).join('')}
                </tbody>
            </table>
        </div>
        <a href="${info.url}" target="_blank" class="source-link">
            <i class="fa-solid fa-arrow-up-right-from-square"></i> Explore Kaggle Dataset
        </a>
    `;
    elements.modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
};

elements.modalClose.onclick = () => {
    elements.modal.classList.add('hidden');
    document.body.style.overflow = 'auto';
};

window.onclick = (e) => { 
    if (e.target === elements.modal) {
        elements.modal.classList.add('hidden');
        document.body.style.overflow = 'auto';
    }
};

elements.tabs.forEach(btn => btn.onclick = () => {
    loadTab(btn.dataset.id);
    btn.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
});

const initialId = elements.tabs[0]?.dataset.id;
if (initialId) loadTab(initialId);
