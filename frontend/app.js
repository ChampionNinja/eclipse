/**
 * Bite.ai — Main Application Logic
 * Handles UI interactions, API calls, session, and profile management.
 */

const API_BASE = `${window.location.protocol}//${window.location.host}`;

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
    voiceCallMode: false,  // continuous voice call
    isSpeaking: false,
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
    tabCamera: $('#tab-camera'),
    tabManual: $('#tab-manual'),
    barcodeCameraPanel: $('#barcode-camera-panel'),
    barcodeManualPanel: $('#barcode-manual-panel'),
    barcodeReader: $('#barcode-reader'),
};

// ===== INIT =====
document.addEventListener('DOMContentLoaded', () => {
    loadProfile();
    generateSessionId();
    bindEvents();

    // Tap the orb to start/stop voice call
    const orbCanvas = document.getElementById('orb-canvas');
    if (orbCanvas) {
        orbCanvas.addEventListener('click', () => toggleVoiceCall());
    }
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

    // Barcode — open modal with camera
    els.btnBarcode.addEventListener('click', openBarcodeModal);
    els.btnCloseBarcode.addEventListener('click', closeBarcodeModal);
    els.btnSubmitBarcode.addEventListener('click', handleBarcode);
    els.inputBarcode.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') handleBarcode();
    });

    // Barcode tabs
    els.tabCamera.addEventListener('click', () => switchBarcodeTab('camera'));
    els.tabManual.addEventListener('click', () => switchBarcodeTab('manual'));

    // Barcode input — digits only
    els.inputBarcode.addEventListener('input', (e) => {
        e.target.value = e.target.value.replace(/\D/g, '').slice(0, 13);
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
            const modal = backdrop.parentElement;
            if (modal === els.barcodeModal) {
                closeBarcodeModal();
            } else {
                closeModal(modal);
            }
        });
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

// ===== BARCODE SCANNER =====
let html5QrCode = null;
let scannerRunning = false;
let availableCameras = [];
let activeCameraId = null;

function openBarcodeModal() {
    openModal(els.barcodeModal);
    switchBarcodeTab('camera');
}

function closeBarcodeModal() {
    stopScanner();
    closeModal(els.barcodeModal);
}

function switchBarcodeTab(tab) {
    if (tab === 'camera') {
        els.tabCamera.classList.add('active');
        els.tabManual.classList.remove('active');
        els.barcodeCameraPanel.classList.remove('hidden');
        els.barcodeManualPanel.classList.add('hidden');
        loadCamerasAndStart();
    } else {
        els.tabCamera.classList.remove('active');
        els.tabManual.classList.add('active');
        els.barcodeCameraPanel.classList.add('hidden');
        els.barcodeManualPanel.classList.remove('hidden');
        stopScanner();
        setTimeout(() => els.inputBarcode.focus(), 100);
    }
}

async function loadCamerasAndStart() {
    if (!window.Html5Qrcode) return;

    try {
        availableCameras = await Html5Qrcode.getCameras();

        if (availableCameras.length === 0) {
            showCameraError('No cameras found — use manual entry');
            return;
        }

        renderCameraSelector();

        const backCam = availableCameras.find(c =>
            /back|rear|environment/i.test(c.label)
        );
        activeCameraId = backCam ? backCam.id : availableCameras[availableCameras.length - 1].id;

        const selector = document.getElementById('camera-selector');
        if (selector) selector.value = activeCameraId;

        await startScannerWithCamera(activeCameraId);
    } catch (err) {
        console.warn('Camera enumeration error:', err);
        showCameraError('Camera access denied — use manual entry');
    }
}

function renderCameraSelector() {
    let selector = document.getElementById('camera-selector');

    if (availableCameras.length <= 1) {
        if (selector) selector.remove();
        return;
    }

    if (!selector) {
        selector = document.createElement('select');
        selector.id = 'camera-selector';
        selector.className = 'camera-selector';
        const reader = document.getElementById('barcode-reader');
        reader.parentNode.insertBefore(selector, reader);
    }

    selector.innerHTML = availableCameras.map(cam => {
        const label = cam.label || `Camera ${availableCameras.indexOf(cam) + 1}`;
        const shortLabel = label.length > 40 ? label.substring(0, 37) + '...' : label;
        return `<option value="${cam.id}">${shortLabel}</option>`;
    }).join('');

    selector.onchange = async () => {
        activeCameraId = selector.value;
        await stopScanner();
        await startScannerWithCamera(activeCameraId);
    };
}

async function startScannerWithCamera(cameraId) {
    if (scannerRunning) await stopScanner();
    if (!window.Html5Qrcode) return;

    try {
        html5QrCode = new Html5Qrcode('barcode-reader');

        const config = {
            fps: 10,
            qrbox: { width: 280, height: 120 },
            formatsToSupport: [
                Html5QrcodeSupportedFormats.EAN_13,
                Html5QrcodeSupportedFormats.EAN_8,
                Html5QrcodeSupportedFormats.UPC_A,
                Html5QrcodeSupportedFormats.UPC_E,
                Html5QrcodeSupportedFormats.CODE_128,
                Html5QrcodeSupportedFormats.CODE_39,
            ],
        };

        await html5QrCode.start(
            cameraId,
            config,
            onScanSuccess,
            () => {}
        );
        scannerRunning = true;

        const hint = els.barcodeCameraPanel.querySelector('.barcode-hint');
        if (hint) {
            hint.textContent = 'Point your camera at a barcode';
            hint.style.color = '';
        }
    } catch (err) {
        console.warn('Camera start error:', err);
        showCameraError('Camera failed to start — try another or use manual entry');
    }
}

function showCameraError(msg) {
    const hint = els.barcodeCameraPanel.querySelector('.barcode-hint');
    if (hint) {
        hint.textContent = msg;
        hint.style.color = 'var(--avoid-color, #e74c3c)';
    }
    setTimeout(() => switchBarcodeTab('manual'), 2000);
}

async function startScanner() {
    await loadCamerasAndStart();
}

async function stopScanner() {
    if (html5QrCode && scannerRunning) {
        try {
            await html5QrCode.stop();
        } catch (e) { /* ignore */ }
        scannerRunning = false;
    }
}

function onScanSuccess(decodedText) {
    // Vibrate if supported
    if (navigator.vibrate) navigator.vibrate(200);

    // Stop scanner immediately
    stopScanner();
    closeBarcodeModal();

    // Process the scanned barcode
    addMessage('user', `📷 Scanned barcode: ${decodedText}`);
    if (nutriOrb) nutriOrb.setState('processing');

    sendQuery(`scan ${decodedText}`)
        .then(result => handleResponse(result))
        .catch(err => {
            addMessage('assistant', 'Couldn\'t connect to the server. Running demo analysis...');
            handleDemoResponse(`scan ${decodedText}`);
        });
}

async function handleBarcode() {
    const barcode = els.inputBarcode.value.trim();
    if (!barcode || barcode.length < 8) {
        els.inputBarcode.style.borderColor = 'var(--avoid-color)';
        setTimeout(() => els.inputBarcode.style.borderColor = '', 1000);
        return;
    }

    closeBarcodeModal();
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

// ===== VOICE AGENT (TTS + STT) =====
let recognition = null;
let selectedVoice = null;

// Pick the best available voice
function initVoice() {
    const voices = speechSynthesis.getVoices();
    // Prefer natural-sounding English voices
    const preferred = ['Google UK English Female', 'Google US English', 'Samantha',
        'Microsoft Zira', 'Microsoft Jenny', 'Karen', 'English Female'];
    for (const name of preferred) {
        const v = voices.find(v => v.name.includes(name));
        if (v) { selectedVoice = v; break; }
    }
    if (!selectedVoice) {
        selectedVoice = voices.find(v => v.lang.startsWith('en')) || voices[0];
    }
}

// Load voices (they load async in Chrome)
if (speechSynthesis.onvoiceschanged !== undefined) {
    speechSynthesis.onvoiceschanged = initVoice;
}
initVoice();

// --- TTS: Speak text aloud ---
function speak(text, onDone) {
    // Cancel any ongoing speech
    speechSynthesis.cancel();

    // Clean text for speech
    const clean = text
        .replace(/[🥗👋🔌📷⚠️]/g, '')
        .replace(/<[^>]+>/g, '')
        .replace(/\s+/g, ' ')
        .trim();

    if (!clean) { if (onDone) onDone(); return; }

    const utterance = new SpeechSynthesisUtterance(clean);
    utterance.rate = 1.05;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;
    if (selectedVoice) utterance.voice = selectedVoice;

    state.isSpeaking = true;
    if (nutriOrb) nutriOrb.setState('speaking');

    utterance.onend = () => {
        state.isSpeaking = false;
        if (onDone) onDone();
    };

    utterance.onerror = () => {
        state.isSpeaking = false;
        if (onDone) onDone();
    };

    speechSynthesis.speak(utterance);
}

function stopSpeaking() {
    speechSynthesis.cancel();
    state.isSpeaking = false;
}

// --- STT: Listen for voice input ---
function startListening() {
    if (!('webkitSpeechRecognition' in window || 'SpeechRecognition' in window)) {
        addMessage('assistant', 'Voice input isn\'t supported in this browser. Try Chrome!');
        return;
    }

    if (state.isRecording) return;

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.lang = 'en-IN';
    recognition.interimResults = false;
    recognition.maxAlternatives = 1;

    recognition.onstart = () => {
        state.isRecording = true;
        els.btnMic.classList.add('recording');
        els.btnMic.querySelector('span').textContent = 'Listening...';
        if (nutriOrb) nutriOrb.setState('listening');
    };

    recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        state.isRecording = false;
        els.btnMic.classList.remove('recording');

        // Show what user said
        els.textInput.value = transcript;
        handleSend();
    };

    recognition.onerror = (event) => {
        console.warn('STT error:', event.error);
        state.isRecording = false;
        els.btnMic.classList.remove('recording');

        if (event.error === 'no-speech' && state.voiceCallMode) {
            // In call mode, just restart listening
            setTimeout(() => startListening(), 300);
        } else if (event.error === 'no-speech') {
            if (nutriOrb) nutriOrb.setState('idle');
        }
    };

    recognition.onend = () => {
        state.isRecording = false;
        els.btnMic.classList.remove('recording');
    };

    recognition.start();
}

function stopListening() {
    state.isRecording = false;
    els.btnMic.classList.remove('recording');
    els.btnMic.querySelector('span').textContent = 'Voice';
    if (recognition) {
        try { recognition.stop(); } catch (e) { /* ignore */ }
    }
}

// --- Voice Call Mode: continuous listen → process → speak → listen ---
function toggleVoiceCall() {
    if (state.voiceCallMode) {
        // End call
        state.voiceCallMode = false;
        stopSpeaking();
        stopListening();
        els.btnMic.querySelector('span').textContent = 'Voice';
        els.btnMic.classList.remove('recording', 'call-active');
        if (nutriOrb) nutriOrb.setState('idle');
        addMessage('assistant', 'Voice call ended. Tap the mic to start again!');
        return;
    }

    // Start call
    state.voiceCallMode = true;
    els.btnMic.classList.add('call-active');
    els.btnMic.querySelector('span').textContent = 'End Call';

    // Greeting
    const greeting = state.userProfile.name
        ? `Hi ${state.userProfile.name}! I'm listening — ask me about any food.`
        : `Hi! I'm listening — ask me about any food.`;

    addMessage('assistant', greeting);
    speak(greeting, () => {
        // After greeting, start listening
        startListening();
    });
}

function toggleVoice() {
    toggleVoiceCall();
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

    // Speak the response (TTS)
    speak(result.response_text, () => {
        // After speaking, update orb to verdict state briefly
        if (nutriOrb && result.verdict) {
            nutriOrb.setState(result.verdict.verdict);
            setTimeout(() => {
                if (!state.isSpeaking && !state.isRecording) {
                    if (state.voiceCallMode) {
                        // In call mode: auto-listen again
                        startListening();
                    } else {
                        nutriOrb.setState('idle');
                    }
                }
            }, 2000);
        } else if (state.voiceCallMode) {
            // Non-verdict response in call mode: keep listening
            startListening();
        } else if (nutriOrb) {
            nutriOrb.setState('idle');
        }
    });
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

    // Source (hidden — no latency display)
    els.responseSource.textContent = '';

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
            response_text: 'Demo mode: Connect the Bite.ai backend for real barcode analysis!',
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
