/* AI Coach front-end. Transport: WebRTC (Pipecat SmallWebRTC) — mic up, bot audio down as an Opus track,
   RTVI JSON events (transcripts, LLM text, speaking state, metrics) on a data channel. */
const $ = (id) => document.getElementById(id);
const els = {
  shell: $('shell'), sidebar: $('sidebar'), navToggle: $('nav-toggle'), navClose: $('nav-close'), scrim: $('scrim'),
  coachList: $('coach-list'), voice: $('voice'), llm: $('llm'), tts: $('tts'), turn: $('turn'), bargein: $('bargein'),
  settings: $('settings'), settingsToggle: $('settings-toggle'), settingsClose: $('settings-close'), ttsNote: $('tts-note'),
  call: $('call'), interrupt: $('interrupt'), mute: $('mute'), muteLabel: $('mute-label'), speaker: $('speaker'), speakerLabel: $('speaker-label'), callControls: $('call-controls'), micMeter: $('mic-meter'), level: $('level'),
  typedForm: $('typed-form'), typed: $('typed'), typedSend: $('typed-send'),
  transcript: $('transcript'), empty: $('empty'),
  dot: $('status-dot'), statusText: $('status-text'), avatarWrap: $('avatar-wrap'), avatar: $('avatar'),
  coachName: $('coach-name'), welcomeAvatar: $('welcome-avatar'), welcomeName: $('welcome-name'), coachTagline: $('coach-tagline'),
  coachMeta: $('coach-meta'), disclaimer: $('disclaimer'), engineLine: $('engine-line'), retrieval: $('retrieval'),
  mStt: $('m-stt'), mRet: $('m-ret'), mLlm: $('m-llm'), mTts: $('m-tts'), mTotal: $('m-total'), audio: $('bot-audio'),
};

const RTVI = 'rtvi-ai';
let config = null, coachId = null;
let pc = null, dc = null, micStream = null, meterCtx = null, meterRaf = null, pingTimer = null;
let inCall = false, connecting = false, currentCoachMsg = null, msgCounter = 0, botSpeaking = false, muted = false;
let stuckTimer = null, stuckWarned = false;

/* ---------------------------------------------------------------- helpers */
const initials = (name) => name.split(/\s+/).map(w => w[0]).join('').slice(0, 2).toUpperCase();
const currentCoach = () => config?.coaches.find(c => c.id === coachId);
const esc = (s) => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));

function setStatus(state, text) {
  els.dot.dataset.state = state;
  els.avatarWrap.dataset.state = state;
  els.statusText.textContent = text || state[0].toUpperCase() + state.slice(1);
}
function idleStatus() {
  if (!inCall) return setStatus('ready', 'Ready');
  if (muted) return setStatus('muted', 'Muted');
  setStatus('listening', micStream ? 'Listening' : 'Ready (typed only)');
}

function addMsg(role, text, cls) {
  els.empty.hidden = true;
  const div = document.createElement('div');
  div.className = `msg ${cls || role}`;
  const who = document.createElement('div'); who.className = 'who-label';
  who.textContent = role === 'me' ? 'You' : role === 'coach' ? (currentCoach()?.name || 'Coach') : '';
  const t = document.createElement('div'); t.className = 'text'; t.textContent = text;
  if (who.textContent) div.appendChild(who);
  div.appendChild(t);
  els.transcript.appendChild(div);
  scrollThread();
  return t;
}
function scrollThread() { els.transcript.scrollTop = els.transcript.scrollHeight; }

/* ---------------------------------------------------------------- sidebar + settings sheet */
const narrow = () => window.innerWidth <= 820;
function setNav(open) {
  els.shell.classList.toggle('nav-closed', !open);
  els.navToggle.setAttribute('aria-expanded', String(open));
  updateScrim();
  try { localStorage.setItem('ai-coach.nav', open ? 'open' : 'closed'); } catch {}
}
function setSettings(open) {
  els.settings.hidden = !open;
  els.settingsToggle.setAttribute('aria-expanded', String(open));
  updateScrim();
}
function updateScrim() {
  const navOpen = !els.shell.classList.contains('nav-closed');
  document.body.classList.toggle('scrim-on', !els.settings.hidden || (navOpen && narrow()));
}
els.navToggle.addEventListener('click', () => setNav(els.shell.classList.contains('nav-closed')));
els.navClose.addEventListener('click', () => setNav(false));
els.settingsToggle.addEventListener('click', () => setSettings(els.settings.hidden));
els.settingsClose.addEventListener('click', () => setSettings(false));
els.scrim.addEventListener('click', () => { setSettings(false); if (narrow()) setNav(false); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { setSettings(false); if (narrow()) setNav(false); } });
window.addEventListener('resize', updateScrim);

function renderCoachList() {
  els.coachList.innerHTML = '';
  for (const c of config.coaches) {
    const b = document.createElement('button');
    b.type = 'button'; b.className = 'coach-item press'; b.dataset.id = c.id;
    b.setAttribute('aria-current', String(c.id === coachId));
    b.disabled = inCall;
    b.innerHTML = `<span class="mini-avatar"></span><span><div class="coach-item-name"></div><div class="coach-item-sub"></div></span>`;
    b.querySelector('.mini-avatar').textContent = initials(c.name);
    b.querySelector('.coach-item-name').textContent = c.name;
    b.querySelector('.coach-item-sub').textContent = c.n_chunks ? `${c.n_sources} sources · ${c.n_chunks} notes` : 'Profile coming soon';
    b.addEventListener('click', () => { if (inCall) return; selectCoach(c.id); if (narrow()) setNav(false); });
    els.coachList.appendChild(b);
  }
}

function selectCoach(id) {
  coachId = id;
  try { localStorage.setItem('ai-coach.coach', id); } catch {}
  for (const b of els.coachList.querySelectorAll('.coach-item')) b.setAttribute('aria-current', String(b.dataset.id === id));
  const c = currentCoach();
  if (!c) return;
  els.coachName.textContent = c.name; els.welcomeName.textContent = c.name;
  els.avatar.textContent = initials(c.name); els.welcomeAvatar.textContent = initials(c.name);
  els.coachTagline.textContent = c.tagline || c.short_bio || '';
  els.coachMeta.textContent = `${c.name}: ${c.n_sources} public sources · ${c.n_chunks} knowledge notes`;
  els.disclaimer.textContent = c.disclaimer || config.disclaimer;
  if (c.tts_engine && [...els.tts.options].some(o => o.value === c.tts_engine)) els.tts.value = c.tts_engine;
  fillVoices(); updateEngineLine(); updateEngineNote();
}

/* ---------------------------------------------------------------- config + settings */
async function loadConfig() {
  const r = await fetch('/api/config');
  config = await r.json();
  let saved = null; try { saved = localStorage.getItem('ai-coach.coach'); } catch {}
  coachId = config.coaches.some(c => c.id === saved) ? saved : config.coaches[0]?.id;
  renderCoachList();
  els.llm.innerHTML = '';
  for (const m of config.llm_models) { const o = document.createElement('option'); o.value = m; o.textContent = `${config.llm_provider} · ${m}`; els.llm.appendChild(o); }
  els.llm.value = config.default_llm_model;
  els.tts.innerHTML = '';
  for (const e of config.tts_engines) { const o = document.createElement('option'); o.value = e.id; o.textContent = e.label; els.tts.appendChild(o); }
  els.tts.value = config.default_tts_engine;
  els.turn.value = config.turn_mode || 'smart';
  els.bargein.checked = !!config.barge_in;
  selectCoach(coachId);
  let nav = 'closed'; try { nav = localStorage.getItem('ai-coach.nav') || (narrow() ? 'closed' : 'open'); } catch {}
  setNav(nav === 'open' && !narrow());
  if (!config.llm_configured) { addMsg('', config.llm_missing_message, 'err'); setStatus('error', 'No LLM key'); }
}

function fillVoices() {
  const engine = els.tts.value;
  const coach = currentCoach();
  const voices = config.voices.filter(v => v.engine === engine);
  els.voice.innerHTML = '';
  for (const v of voices) {
    const o = document.createElement('option'); o.value = v.id;
    o.textContent = v.kind === 'custom' ? `★ ${v.name} (custom voice)` : v.name;
    els.voice.appendChild(o);
  }
  const pref = coach?.voice?.[engine];
  if (pref && voices.some(v => v.id === pref)) els.voice.value = pref;
  else if (config.default_voice[engine] && voices.some(v => v.id === config.default_voice[engine])) els.voice.value = config.default_voice[engine];
}
function updateEngineNote() {
  const e = config.tts_engines.find(x => x.id === els.tts.value);
  els.ttsNote.textContent = e?.note || '';
  els.ttsNote.classList.toggle('warn', els.tts.value === 'breeze');
}
function updateEngineLine() {
  els.engineLine.textContent = `stt ${config.stt_engine} · tts ${els.tts.value} · llm ${config.llm_provider} / ${els.llm.value} · turn ${els.turn.value}`;
}
els.tts.addEventListener('change', () => { fillVoices(); updateEngineLine(); updateEngineNote(); });
els.llm.addEventListener('change', updateEngineLine);
els.turn.addEventListener('change', updateEngineLine);

/* ---------------------------------------------------------------- stuck-call watchdog (Breeze on a weak Mac) */
function armStuckWatchdog() {
  if (els.tts.value !== 'breeze' || stuckWarned) return;
  clearTimeout(stuckTimer);
  stuckTimer = setTimeout(() => {
    if (!inCall || stuckWarned) return;
    stuckWarned = true;
    addMsg('', 'The call looks stuck: Breeze is taking too long on this computer. Your Mac is not strong enough to run it live. End the call and pick another speech engine, for example Pocket TTS, in Settings.', 'warn');
  }, 15000);
}
function disarmStuckWatchdog() { clearTimeout(stuckTimer); stuckTimer = null; }

/* ---------------------------------------------------------------- mute */
function setSpeakerMuted(on) {
  els.audio.muted = on;
  els.speaker.setAttribute('aria-pressed', String(on));
  els.speakerLabel.textContent = on ? 'Unmute coach' : 'Mute coach';
  els.speaker.title = on ? "Hear the coach again (S)" : "Mute the coach's voice (S)";
}
function setMuted(on) {
  muted = on;
  for (const t of micStream?.getAudioTracks() || []) t.enabled = !on;   // WebRTC sends silence while disabled
  els.mute.setAttribute('aria-pressed', String(on));
  els.muteLabel.textContent = on ? 'Unmute mic' : 'Mute mic';
  els.mute.title = on ? 'Unmute your microphone (M)' : 'Mute your microphone (M)';
  els.micMeter.classList.toggle('muted', on);
  if (on) els.level.style.width = '0%';
  if (inCall && !botSpeaking) idleStatus();
}
els.mute.addEventListener('click', () => setMuted(!muted));
els.speaker.addEventListener('click', () => setSpeakerMuted(!els.audio.muted));
document.addEventListener('keydown', (e) => {
  if (['INPUT', 'TEXTAREA', 'SELECT'].includes(document.activeElement?.tagName) || !inCall) return;
  if (e.key.toLowerCase() === 'm' && micStream) { e.preventDefault(); setMuted(!muted); }
  if (e.key.toLowerCase() === 's') { e.preventDefault(); setSpeakerMuted(!els.audio.muted); }
});

/* ---------------------------------------------------------------- WebRTC / RTVI */
function sendRtvi(type, data) {
  if (dc?.readyState !== 'open') return;
  dc.send(JSON.stringify({ label: RTVI, type, id: `m${++msgCounter}`, data }));
}
const sendClientMessage = (t, d) => sendRtvi('client-message', { t, d });

async function waitIceComplete(pc) {
  if (pc.iceGatheringState === 'complete') return;
  await new Promise((resolve) => {
    const check = () => { if (pc.iceGatheringState === 'complete') { pc.removeEventListener('icegatheringstatechange', check); resolve(); } };
    pc.addEventListener('icegatheringstatechange', check);
    setTimeout(resolve, 1500);
  });
}

function startMeter(stream) {
  try {
    meterCtx = new AudioContext();
    const src = meterCtx.createMediaStreamSource(stream);
    const an = meterCtx.createAnalyser(); an.fftSize = 512; src.connect(an);
    const buf = new Float32Array(an.fftSize);
    const tick = () => { an.getFloatTimeDomainData(buf); let s = 0; for (const v of buf) s += v * v;
      if (!muted) els.level.style.width = `${Math.min(100, Math.sqrt(s / buf.length) * 450)}%`; meterRaf = requestAnimationFrame(tick); };
    tick();
  } catch {}
}

function setCallUi(on) {
  inCall = on; connecting = false;
  els.call.dataset.state = on ? 'live' : 'idle';
  els.call.setAttribute('aria-label', on ? 'End call' : 'Start call');
  els.call.disabled = false;
  els.callControls.hidden = !on;
  els.mute.disabled = !micStream; els.mute.title = micStream ? 'Mute your microphone (M)' : 'No microphone in this call';
  els.micMeter.hidden = !micStream;
  els.interrupt.disabled = true;
  if (!on) setSpeakerMuted(false);
  [els.voice, els.llm, els.tts, els.turn, els.bargein].forEach(s => s.disabled = on);
  for (const b of els.coachList.querySelectorAll('.coach-item')) b.disabled = on;
  if (!on) setMuted(false);
  idleStatus();
}

async function startCall() {
  if (connecting) return;
  connecting = true; els.call.dataset.state = 'connecting'; els.call.disabled = true;
  setStatus('thinking', 'Connecting…');
  try {
    micStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 } });
    startMeter(micStream);
  } catch (e) {
    micStream = null;
    addMsg('', `No microphone (${e.message}). You can still type to the coach and hear the answers.`, 'warn');
  }
  pc = new RTCPeerConnection({ iceServers: [] });
  dc = pc.createDataChannel('chat', { ordered: true });
  dc.onopen = () => {
    sendRtvi('client-ready', { version: '1.0.0', about: { library: 'ai-coach-web', library_version: '0.3.0' } });
    pingTimer = setInterval(() => { if (dc?.readyState === 'open') dc.send('ping'); }, 1000);
  };
  dc.onmessage = (e) => { if (typeof e.data === 'string' && e.data.startsWith('pong')) return; try { onRtvi(JSON.parse(e.data)); } catch {} };
  if (micStream) { for (const track of micStream.getAudioTracks()) pc.addTransceiver(track, { direction: 'sendrecv' }); }
  else pc.addTransceiver('audio', { direction: 'recvonly' });
  pc.ontrack = (e) => { els.audio.srcObject = e.streams[0]; els.audio.play().catch(() => {}); };
  pc.onconnectionstatechange = () => {
    if (['failed', 'disconnected', 'closed'].includes(pc?.connectionState) && inCall) { addMsg('', 'Connection lost.', 'sys'); teardown(); }
  };
  const offer = await pc.createOffer();
  await pc.setLocalDescription(offer);
  await waitIceComplete(pc);
  const request_data = { coach: coachId, voice: els.voice.value, llm_model: els.llm.value, tts_engine: els.tts.value,
    barge_in: els.bargein.checked, turn_mode: els.turn.value };
  let answer;
  try {
    const r = await fetch('/api/offer', { method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type, request_data }) });
    if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
    answer = await r.json();
  } catch (e) { addMsg('', `Could not start the call: ${e.message}`, 'err'); teardown(); setStatus('error', 'Server error'); return; }
  await pc.setRemoteDescription({ sdp: answer.sdp, type: answer.type });
  setSettings(false);
  setCallUi(true);
  stuckWarned = false;
  if (els.tts.value === 'breeze') addMsg('', 'Breeze gives the best voice but needs a strong computer. If the call gets stuck or the coach pauses for a long time between sentences, end the call and pick Pocket TTS as the speech engine.', 'warn');
}

function onRtvi(m) {
  if (m.label !== RTVI) return;
  const d = m.data || {};
  switch (m.type) {
    case 'bot-ready': idleStatus(); break;
    case 'server-message': onServerMessage(d); break;
    case 'user-started-speaking': setStatus('hearing', 'Hearing you'); break;
    case 'user-stopped-speaking': setStatus('thinking', 'Thinking'); break;
    case 'user-transcription':
      if (d.final) { addMsg('me', d.text); if (!currentCoachMsg) { currentCoachMsg = addMsg('coach', ''); currentCoachMsg.classList.add('cursor'); } }
      break;
    case 'bot-llm-started':
      if (!currentCoachMsg) { currentCoachMsg = addMsg('coach', ''); currentCoachMsg.classList.add('cursor'); }
      setStatus('thinking', 'Thinking'); armStuckWatchdog(); break;
    case 'bot-llm-text':
      if (currentCoachMsg) { currentCoachMsg.textContent += d.text; scrollThread(); }
      break;
    case 'bot-started-speaking': botSpeaking = true; disarmStuckWatchdog(); els.interrupt.disabled = false; setStatus('speaking', 'Speaking'); break;
    case 'bot-stopped-speaking':
      botSpeaking = false; els.interrupt.disabled = true;
      if (currentCoachMsg) currentCoachMsg.classList.remove('cursor'); currentCoachMsg = null; idleStatus(); break;
    case 'bot-interrupted':
      if (currentCoachMsg && currentCoachMsg.textContent) { currentCoachMsg.classList.remove('cursor'); currentCoachMsg.textContent += ' […interrupted]'; currentCoachMsg = null; }
      break;
    case 'metrics': onMetrics(d); break;
    case 'error':
      addMsg('', friendlyError(d.message || d.error || JSON.stringify(d)), 'err');
      if (d.fatal || /API key|api_key|401|403/i.test(String(d.message || d.error))) setStatus('error', 'LLM error');
      break;
  }
}
function friendlyError(s) {
  if (/Incorrect API key|invalid_api_key|401/i.test(s)) return 'The LLM rejected the API key. Check .env (XAI_API_KEY / GROQ_API_KEY) and restart the server.';
  if (/rate limit|429/i.test(s)) return 'The LLM provider is rate-limiting requests. Wait a moment and try again.';
  if (/no audio/i.test(s)) return 'The speech engine produced no audio for a fragment; continuing.';
  return s;
}
function onServerMessage(d) {
  if (d.t === 'session') {
    if (!d.llm_configured) addMsg('', d.llm_missing_message, 'err');
  } else if (d.t === 'retrieval') {
    els.retrieval.innerHTML = '';
    for (const c of d.chunks) {
      const div = document.createElement('div'); div.className = 'chunk';
      div.innerHTML = `<span class="k">${esc(c.topic)}</span> · ${esc(c.evidence.toLowerCase())} · ${c.score} · ${esc(c.text)}`;
      els.retrieval.appendChild(div);
    }
    els.mRet.textContent = d.ms;
  } else if (d.t === 'latency') {
    if (d.stt_ms != null) els.mStt.textContent = d.stt_ms;
    if (d.llm_first_token_ms != null) els.mLlm.textContent = d.llm_first_token_ms;
    if (d.tts_first_audio_ms != null) els.mTts.textContent = d.tts_first_audio_ms;
    if (d.total_ms != null) els.mTotal.textContent = d.total_ms;
  }
}
function onMetrics(d) {
  for (const p of d.processing || []) if (/STT/i.test(p.processor) && p.value) els.mStt.textContent = Math.round(p.value * 1000);
}

function teardown() {
  botSpeaking = false; disarmStuckWatchdog();
  clearInterval(pingTimer); pingTimer = null;
  if (meterRaf) cancelAnimationFrame(meterRaf); meterRaf = null;
  try { meterCtx?.close(); } catch {}
  try { micStream?.getTracks().forEach(t => t.stop()); } catch {}
  try { dc?.close(); } catch {}
  try { pc?.close(); } catch {}
  pc = null; dc = null; micStream = null; meterCtx = null; currentCoachMsg = null;
  els.audio.srcObject = null;
  els.level.style.width = '0%';
  setCallUi(false);
}

els.call.addEventListener('click', () => { if (inCall) { addMsg('', 'Call ended.', 'sys'); teardown(); } else startCall(); });
els.interrupt.addEventListener('click', () => sendClientMessage('interrupt', {}));
els.typed.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); els.typedForm.requestSubmit(); } });
els.typedForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = els.typed.value.trim();
  if (!text) return;
  if (!inCall) { await startCall(); if (!inCall) return; await new Promise(r => setTimeout(r, 600)); }
  if (dc?.readyState !== 'open') return;
  addMsg('me', text); currentCoachMsg = addMsg('coach', ''); currentCoachMsg.classList.add('cursor');
  setStatus('thinking', 'Thinking');
  sendClientMessage('text', { text });
  els.typed.value = '';
});

loadConfig().catch(e => addMsg('', `Failed to load config: ${e.message}`, 'err'));
