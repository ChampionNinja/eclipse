/**
 * NutriAssist — Main Application Logic
 * Handles UI interactions, API calls, session, and profile management.
 */

const API_BASE = 'http://localhost:8000'; // Change for Colab: ngrok URL

// ===== STATE =====
const state = {
    sessionId: null,
    userProfile: {
        name: '',
        age: '',
        gender: '',
        diet_type: '',
        allergies: [],
        conditions: [],
    },
    isRecording: false,
    currentProduct: null,
};

// ===== DOM REFS =====
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

const els = {
    textInput: $('#text-input'),
    btnSend: $('#btn-send'),
    btnMic: $('#btn-mic'),
    btnBarcode: $('#btn-barcode'),
    btnProfile: $('#btn-profile'),
    btnHistory: $('#btn-history'),
    chatHistory: $('#chat-history'),
    responseCard: $('#response-card'),
    productName: $('#response-product-name'),
    verdictBadge: $('#response-badge'),
    responseReason: $('#response-reason'),
    confidenceValue: $('#confidence-value'),
    confidenceFill: $('#confidence-fill'),
    responseSource: $('#response-source'),
    // Profile modal
    profileModal: $('#profile-modal'),
    btnCloseProfile: $('#btn-close-profile'),
    btnSaveProfile: $('#btn-save-profile'),
    inputName: $('#input-name'),
    inputAge: $('#input-age'),
    inputGender: $('#input-gender'),
    inputDiet: $('#input-diet'),
    allergyOptions: $('#allergy-options'),
    conditionOptions: $('#condition-options'),
    // Barcode modal
    barcodeModal: $('#barcode-modal'),
    btnCloseBarcode: $('#btn-close-barcode'),
    btnSubmitBarcode: $('#btn-submit-barcode'),
    inputBarcode: $('#input-barcode'),
};

// ===== INIT =====
document.addEventListener('DOMContentLoaded', () => {
    loadProfile();
    generateSessionId();
    bindEvents();
});

function generateSessionId() {
    state.sessionId = 'sess_' + Date.now() + '_' + Math.random().toString(36).substr(2, 6);
}

// ===== PROFILE (localStorage) =====
function loadProfile() {
    try {
        const saved = localStorage.getItem('nutriassist_profile');
        if (saved) {
            const profile = JSON.parse(saved);
            state.userProfile = { ...state.userProfile, ...profile };
            els.inputName.value = profile.name || '';
            els.inputAge.value = profile.age || '';
            els.inputDiet.value = profile.diet_type || '';
            els.inputGender.value = profile.gender || '';
            renderConditions(profile.gender || '');
            applyProfileTheme(profile.gender || '');
            restoreCheckboxes('allergy', profile.allergies || []);
            restoreCheckboxes('condition', profile.conditions || []);
            return;
        }
        renderConditions('');
        applyProfileTheme('');
    } catch (e) {
        console.warn('Could not load profile:', e);
    }
}

function renderConditions(gender) {
    const container = els.conditionOptions;
    const baseConditions = ['Diabetes', 'Hypertension', 'High Cholesterol'];
    const femaleConditions = ['PCOS', 'Menopause', 'Diabetes', 'Hypertension', 'High Cholesterol'];
    const list = gender === 'female' ? femaleConditions : baseConditions;

    container.innerHTML = list.map(c =>
        `<label class="checkbox-label">
      <span>${c}</span>
      <input type="checkbox" name="condition" value="${c.toLowerCase().replace(/\s/g, '_')}">
    </label>`
    ).join('');

    restoreCheckboxes('condition', state.userProfile.conditions || []);
}

function applyProfileTheme(gender) {
    if (!els.profileModal) return;
    els.profileModal.classList.toggle('female-theme', gender === 'female');
}

function restoreCheckboxes(name, savedValues) {
    document.querySelectorAll(`input[name="${name}"]`).forEach(cb => {
        cb.checked = savedValues.includes(cb.value);
    });
}

function saveProfile() {
    const name = els.inputName.value.trim();
    const age = els.inputAge.value.trim();
    const diet_type = els.inputDiet.value;
    const gender = els.inputGender.value;

    const allergies = [...document.querySelectorAll('input[name="allergy"]:checked')]
        .map(cb => cb.value);
    const conditions = [...document.querySelectorAll('input[name="condition"]:checked')]
        .map(cb => cb.value);

    state.userProfile = { name, age, gender, diet_type, allergies, conditions };

    try {
        localStorage.setItem('nutriassist_profile', JSON.stringify(state.userProfile));
    } catch (e) {
        console.warn('Could not save profile:', e);
    }

    closeModal(els.profileModal);
    addMessage('assistant', `Profile updated! ${name ? 'Hi ' + name + '! ' : ''}${allergies.length ? 'Allergies noted: ' + allergies.join(', ') + '. ' : ''}I'll keep your preferences in mind.`);
}

// ===== EVENTS =====
function bindEvents() {
    // Send text
    els.btnSend.addEventListener('click', handleSend);
    els.textInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSend();
        }
    });

    // Voice
    els.btnMic.addEventListener('click', toggleVoice);

    // Barcode
    els.btnBarcode.addEventListener('click', () => openModal(els.barcodeModal));
    els.btnCloseBarcode.addEventListener('click', () => closeModal(els.barcodeModal));
    els.btnSubmitBarcode.addEventListener('click', handleBarcode);
    els.inputBarcode.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') handleBarcode();
    });

    // Profile
    els.btnProfile.addEventListener('click', () => openModal(els.profileModal));
    els.btnCloseProfile.addEventListener('click', () => closeModal(els.profileModal));
    els.btnSaveProfile.addEventListener('click', saveProfile);
    els.inputGender.addEventListener('change', (e) => {
        renderConditions(e.target.value);
        applyProfileTheme(e.target.value);
    });

    // Modal backdrop close
    $$('.modal-backdrop').forEach(backdrop => {
        backdrop.addEventListener('click', () => {
            closeModal(backdrop.parentElement);
        });
    });

    // Barcode input — digits only
    els.inputBarcode.addEventListener('input', (e) => {
        e.target.value = e.target.value.replace(/\D/g, '').slice(0, 13);
    });
}

// ===== SEND QUERY =====
async function handleSend() {
    const text = els.textInput.value.trim();
    if (!text) return;

    els.textInput.value = '';
    addMessage('user', text);

    // Set orb to processing
    if (nutriOrb) nutriOrb.setState('processing');

    try {
        const result = await sendQuery(text);
        handleResponse(result);
    } catch (err) {
        addMessage('assistant', 'Sorry, I couldn\'t reach the server. Make sure the backend is running! 🔌');
        if (nutriOrb) nutriOrb.setState('idle');
        // Demo mode fallback
        handleDemoResponse(text);
    }
}

// ===== BARCODE =====
async function handleBarcode() {
    const barcode = els.inputBarcode.value.trim();
    if (!barcode || barcode.length < 8) {
        els.inputBarcode.style.borderColor = 'var(--avoid-color)';
        setTimeout(() => els.inputBarcode.style.borderColor = '', 1000);
        return;
    }

    closeModal(els.barcodeModal);
    els.inputBarcode.value = '';

    addMessage('user', `Scan barcode: ${barcode}`);
    if (nutriOrb) nutriOrb.setState('processing');

    try {
        const result = await sendQuery(`scan ${barcode}`);
        handleResponse(result);
    } catch (err) {
        addMessage('assistant', 'Couldn\'t connect to the server. Running demo analysis...');
        handleDemoResponse(`scan ${barcode}`);
    }
}

// ===== VOICE =====
let recognition = null;

function toggleVoice() {
    if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
        addMessage('assistant', 'Voice input isn\'t supported in this browser. Try Chrome!');
        return;
    }

    if (state.isRecording) {
        stopVoice();
        return;
    }

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.lang = 'en-IN';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
        state.isRecording = true;
        els.btnMic.classList.add('recording');
        els.btnMic.querySelector('span').textContent = 'Stop';
        if (nutriOrb) nutriOrb.setState('listening');
    };

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        els.textInput.value = transcript;
        stopVoice();
        handleSend();
    };

    recognition.onerror = (event) => {
        console.error('Speech error:', event.error);
        stopVoice();
        if (event.error === 'no-speech') {
            addMessage('assistant', 'I didn\'t hear anything. Try again?');
        }
    };

    recognition.onend = () => stopVoice();

    recognition.start();
}

function stopVoice() {
    state.isRecording = false;
    els.btnMic.classList.remove('recording');
    els.btnMic.querySelector('span').textContent = 'Voice';
    if (recognition) {
        try { recognition.stop(); } catch (e) { /* ignore */ }
    }
    if (nutriOrb && nutriOrb.state === 'listening') {
        nutriOrb.setState('idle');
    }
}

// ===== API CALL =====
async function sendQuery(text) {
    const resp = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            text: text,
            session_id: state.sessionId,
            user_profile: state.userProfile,
        }),
    });

    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return await resp.json();
}

// ===== HANDLE RESPONSE =====
function handleResponse(result) {
    // Update session
    state.sessionId = result.session_id || state.sessionId;

    // Add chat message
    addMessage('assistant', result.response_text);

    // Update response card if there's a verdict
    if (result.verdict) {
        showResponseCard(result);
    }

    // Update orb
    if (nutriOrb) {
        const orbState = result.verdict ? result.verdict.verdict : 'idle';
        nutriOrb.setState(orbState);
        // Reset to idle after 5 seconds
        setTimeout(() => {
            if (nutriOrb.state === orbState && orbState !== 'idle') {
                nutriOrb.setState('idle');
            }
        }, 5000);
    }
}

function showResponseCard(result) {
    const verdict = result.verdict;
    const card = els.responseCard;

    // Product name
    els.productName.textContent = result.product_name || 'Unknown Product';

    // Verdict badge
    els.verdictBadge.textContent = verdict.verdict;
    els.verdictBadge.className = 'verdict-badge ' + verdict.verdict;

    // Reason
    els.responseReason.textContent = verdict.reason;

    // Confidence
    const confPct = Math.round(verdict.confidence * 100);
    els.confidenceValue.textContent = confPct + '%';
    els.confidenceFill.className = 'confidence-fill ' + verdict.verdict;

    // Animate confidence bar
    setTimeout(() => {
        els.confidenceFill.style.width = confPct + '%';
    }, 50);

    // Source
    els.responseSource.textContent = `Source: ${result.intent || 'unknown'} · ${Math.round(result.latency_ms || 0)}ms`;

    // Show card with state class
    card.className = 'response-card ' + verdict.verdict;
    card.classList.remove('hidden');
}

// ===== DEMO MODE (no backend) =====
function handleDemoResponse(text) {
    const textLower = text.toLowerCase();

    // Simple demo: analyze based on keywords
    let demoResult = null;

    if (textLower.includes('oat') || textLower.includes('dal') || textLower.includes('brown rice') || textLower.includes('salad')) {
        demoResult = {
            product_name: extractFoodName(text),
            intent: 'food_query',
            verdict: { verdict: 'eat', reason: 'High fiber, good protein content, nutrient-dense food', confidence: 0.88 },
            response_text: `${extractFoodName(text)}: Go ahead! High fiber, good protein content, nutrient-dense food. Want to know more?`,
            latency_ms: 340,
        };
    } else if (textLower.includes('cola') || textLower.includes('soda') || textLower.includes('chips') || textLower.includes('maggi') || textLower.includes('nutella')) {
        demoResult = {
            product_name: extractFoodName(text),
            intent: 'food_query',
            verdict: { verdict: 'avoid', reason: 'High sugar/sodium, heavily processed, low nutritional value', confidence: 0.92 },
            response_text: `${extractFoodName(text)}: I'd skip this one. High sugar/sodium, heavily processed, low nutritional value. Ask me about alternatives!`,
            latency_ms: 280,
        };
    } else if (textLower.includes('bread') || textLower.includes('rice') || textLower.includes('paneer') || textLower.includes('egg')) {
        demoResult = {
            product_name: extractFoodName(text),
            intent: 'food_query',
            verdict: { verdict: 'sometimes', reason: 'Moderate nutrition profile; portion control recommended', confidence: 0.75 },
            response_text: `${extractFoodName(text)}: It's okay in moderation. Moderate nutrition profile; portion control recommended. Want details?`,
            latency_ms: 310,
        };
    } else if (textLower.match(/\d{8,13}/)) {
        demoResult = {
            product_name: 'Demo Product',
            intent: 'barcode',
            verdict: { verdict: 'sometimes', reason: 'Demo mode — connect backend for real analysis', confidence: 0.5 },
            response_text: 'Demo mode: Connect the NutriAssist backend for real barcode analysis!',
            latency_ms: 100,
        };
    } else if (textLower.includes('hello') || textLower.includes('hi') || textLower.includes('hey')) {
        addMessage('assistant', 'Hey there! 👋 Ask me about any food — like "is paneer healthy?" or scan a barcode!');
        if (nutriOrb) nutriOrb.setState('idle');
        return;
    } else {
        // Generic food query
        demoResult = {
            product_name: extractFoodName(text),
            intent: 'food_query',
            verdict: { verdict: 'sometimes', reason: 'Limited data available — try being more specific or scan a barcode', confidence: 0.45 },
            response_text: `I need more data to fully analyze "${extractFoodName(text)}". Try scanning a barcode for detailed analysis!`,
            latency_ms: 150,
        };
    }

    // Check allergies in demo mode
    if (demoResult && state.userProfile.allergies.length > 0) {
        const foodLower = (demoResult.product_name || '').toLowerCase();
        const hasAllergen = state.userProfile.allergies.some(a =>
            foodLower.includes(a) || textLower.includes(a)
        );
        if (hasAllergen) {
            demoResult.verdict = {
                verdict: 'avoid',
                reason: `Contains ingredients matching your allergies (${state.userProfile.allergies.join(', ')})`,
                confidence: 0.99,
            };
            demoResult.response_text = `⚠️ ${demoResult.product_name}: AVOID — matches your allergy profile!`;
        }
    }

    if (demoResult) {
        // Simulate processing delay
        setTimeout(() => handleResponse(demoResult), 600);
    }
}

function extractFoodName(text) {
    let cleaned = text.toLowerCase()
        .replace(/^(is|are|can i eat|should i eat|tell me about|what about|how healthy is|scan)\s+/i, '')
        .replace(/\s*(healthy|good|bad|safe|ok|for me|for health|for weight loss|for diabetes)\s*\??$/i, '')
        .replace(/\d{8,13}/g, '')
        .trim();
    return cleaned ? cleaned.charAt(0).toUpperCase() + cleaned.slice(1) : 'Unknown Food';
}

// ===== CHAT =====
function addMessage(role, content) {
    const div = document.createElement('div');
    div.className = `chat-message ${role}`;

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.innerHTML = content;

    div.appendChild(bubble);
    els.chatHistory.appendChild(div);

    // Auto-scroll
    requestAnimationFrame(() => {
        els.chatHistory.scrollTop = els.chatHistory.scrollHeight;
    });
}

// ===== MODALS =====
function openModal(modal) {
    modal.classList.remove('hidden');
    // Focus first input
    const input = modal.querySelector('input');
    if (input) setTimeout(() => input.focus(), 100);
}

function closeModal(modal) {
    modal.classList.add('hidden');
}
