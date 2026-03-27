// Genesis Mind V7 — Dashboard Application Logic

let emotionChart = null;
let drivesChart = null;

// UI Elements
const phaseBadge = document.getElementById('phase-badge');
const threadsContainer = document.getElementById('threads-container');
const sensesContainer = document.getElementById('senses-container');
const wmContainer = document.getElementById('wm-container');
const wmUsage = document.getElementById('wm-usage');
const wmCapacity = document.getElementById('wm-capacity');
const hiddenStateContainer = document.getElementById('hidden-state-container');
const activityStream = document.getElementById('activity-stream');
const nodeCount = document.getElementById('node-count');

// Vis Network configuration
let network = null;
let nodesData = new vis.DataSet([]);
let edgesData = new vis.DataSet([]);
const networkContainer = document.getElementById('network-canvas');

const brainOptions = {
    nodes: {
        shape: 'dot',
        borderWidth: 0,
        font: { color: 'rgba(255,255,255,0.7)', size: 12 }
    },
    edges: {
        width: 1,
        color: 'rgba(255,255,255,0.08)',
        smooth: { type: 'cubicBezier' }
    },
    groups: {
        label: { shape: 'text', font: {size: 20} },
        senses: { size: 20, color: {background: 'rgba(239, 83, 80, 0.8)'} },
        limbic: { size: 20, color: {background: 'rgba(255, 152, 0, 0.8)'} },
        memory: { size: 15, color: {background: 'rgba(186, 104, 200, 0.8)'} },
        hidden: { size: 10, color: {background: 'rgba(79, 195, 247, 0.3)'} },
        output: { size: 14, color: {background: 'rgba(255, 213, 79, 0.7)'} },
        acoustic: { size: 18, color: {background: 'rgba(0, 230, 118, 0.8)'} },
        vocoder: { size: 16, color: {background: 'rgba(124, 77, 255, 0.8)'} }
    },
    physics: false,
    interaction: { zoomView: true, dragView: true }
};

function initNetwork() {
    if (!networkContainer) return;
    
    // Build the V7 Global Neural Architecture
    let nodes = [];
    let edges = [];
    
    // Region Labels
    nodes.push({id: 'label_senses', label: 'SENSES\nVision/Audio/Body', group: 'label', x: -800, y: -250, font: {size: 16, color: '#fff'}});
    nodes.push({id: 'label_limbic', label: 'LIMBIC SYSTEM\nEmotions & Drives', group: 'label', x: -500, y: -250, font: {size: 16, color: '#fff'}});
    nodes.push({id: 'label_memory', label: 'MEMORY & DREAMS\nSemantic & Episodic', group: 'label', x: -150, y: -250, font: {size: 16, color: '#fff'}});
    nodes.push({id: 'label_hidden', label: 'SUBCONSCIOUS CORE\n128-dim GRU', group: 'label', x: 250, y: -250, font: {size: 16, color: '#fff'}});
    nodes.push({id: 'label_world', label: 'WORLD MODEL\nPredictions', group: 'label', x: 650, y: -250, font: {size: 16, color: '#fff'}});
    nodes.push({id: 'label_acoustic', label: 'V7: ACOUSTIC PIPELINE\nHear → Think → Speak', group: 'label', x: -800, y: 200, font: {size: 16, color: '#00e676'}});

    // 1. Sensory Input Layer
    nodes.push({id: 'sens_vision', label: 'Vision', group: 'senses', x: -800, y: -100});
    nodes.push({id: 'sens_audio', label: 'Audio', group: 'senses', x: -800, y: 0});
    nodes.push({id: 'sens_body', label: 'Body', group: 'senses', x: -800, y: 100});
    
    // 2. Limbic System Layer
    nodes.push({id: 'limbic_vta', label: 'VTA (Reward)', group: 'limbic', x: -500, y: -100});
    nodes.push({id: 'limbic_amygdala', label: 'Amygdala (Fear)', group: 'limbic', x: -500, y: 0});
    nodes.push({id: 'limbic_pineal', label: 'Pineal (Sleep)', group: 'limbic', x: -500, y: 100});
    
    // Wiring Senses -> Limbic
    edges.push({from: 'sens_vision', to: 'limbic_vta'});
    edges.push({from: 'sens_audio', to: 'limbic_amygdala'});
    edges.push({from: 'sens_body', to: 'limbic_pineal'});
    edges.push({from: 'sens_body', to: 'limbic_vta'});
    
    // 3. Hidden State (128 nodes, 16x8 grid)
    let hIdx = 0;
    for (let col = 0; col < 16; col++) {
        for (let row = 0; row < 8; row++) {
            nodes.push({
                id: `h_${hIdx}`, 
                label: '', 
                group: 'hidden', 
                x: 100 + (col * 25), 
                y: (row * 30) - 105,
                title: `Hidden Activation (GRU Unit ${hIdx})`
            });
            hIdx++;
        }
    }
    
    // Wiring Limbic -> Hidden
    for (let row = 0; row < 8; row++) {
        edges.push({from: 'limbic_vta', to: `h_${row}`});
        edges.push({from: 'limbic_amygdala', to: `h_${row}`});
        edges.push({from: 'limbic_pineal', to: `h_${row}`});
    }

    // Wiring Hidden -> Hidden (Dense-like)
    for (let col = 0; col < 15; col++) {
        for (let row = 0; row < 8; row++) {
            if (Math.random() > 0.4) {
                edges.push({from: `h_${col*8 + row}`, to: `h_${(col+1)*8 + Math.floor(Math.random()*8)}`});
            }
        }
    }
    
    // 4. World Model / Output
    for (let i = 0; i < 8; i++) {
        nodes.push({id: `out_${i}`, label: `P_${i}`, group: 'output', x: 650, y: (i * 30) - 105, title: `Prediction Node ${i}`});
    }

    // Wiring Hidden -> Output
    for (let i = 0; i < 8; i++) {
        for (let row = 0; row < 8; row++) {
            if (Math.random() > 0.6) edges.push({from: `h_${15*8 + row}`, to: `out_${i}`});
        }
    }

    // ═══════════════════════════════════════════
    // V7: ACOUSTIC NEURAL PIPELINE (bottom row)
    // ═══════════════════════════════════════════
    
    // Auditory Cortex (Mel Encoder)
    nodes.push({id: 'ac_mic', label: '🎤 Mic', group: 'senses', x: -800, y: 300, title: 'Microphone Input (16kHz)'});
    nodes.push({id: 'ac_mel', label: 'Mel Filter\n80 bands', group: 'acoustic', x: -600, y: 300, title: 'Mel Spectrogram (138K params)'});
    nodes.push({id: 'ac_enc', label: 'Conv1D\nEncoder', group: 'acoustic', x: -400, y: 300, title: 'Auditory Cortex Encoder → 64-dim'});
    
    // VQ Codebook
    nodes.push({id: 'ac_vq', label: 'VQ Codebook\n256 Phonemes', group: 'acoustic', x: -150, y: 300, title: 'Vector Quantized Codebook (16K params)'});
    
    // Acoustic Transformer (4 layers displayed)
    for (let i = 0; i < 4; i++) {
        nodes.push({
            id: `ac_tf_${i}`, 
            label: `TF Layer ${i+1}`, 
            group: 'acoustic', 
            x: 100 + (i * 100), 
            y: 300,
            title: `Transformer Layer ${i+1} (4 heads, 128-dim)`
        });
    }
    
    // Neural Vocoder
    nodes.push({id: 'ac_recon', label: 'Mel Recon', group: 'vocoder', x: 550, y: 300, title: 'Mel Reconstructor (130K params)'});
    nodes.push({id: 'ac_gl', label: 'Griffin-Lim', group: 'vocoder', x: 700, y: 300, title: 'Phase Reconstruction'});
    nodes.push({id: 'ac_spk', label: '🔊 Speaker', group: 'senses', x: 850, y: 300, title: 'Audio Output'});
    
    // Wire acoustic pipeline
    edges.push({from: 'ac_mic', to: 'ac_mel', color: 'rgba(0, 230, 118, 0.4)'});
    edges.push({from: 'ac_mel', to: 'ac_enc', color: 'rgba(0, 230, 118, 0.4)'});
    edges.push({from: 'ac_enc', to: 'ac_vq', color: 'rgba(0, 230, 118, 0.4)'});
    edges.push({from: 'ac_vq', to: 'ac_tf_0', color: 'rgba(0, 230, 118, 0.4)'});
    edges.push({from: 'ac_tf_0', to: 'ac_tf_1', color: 'rgba(0, 230, 118, 0.4)'});
    edges.push({from: 'ac_tf_1', to: 'ac_tf_2', color: 'rgba(0, 230, 118, 0.4)'});
    edges.push({from: 'ac_tf_2', to: 'ac_tf_3', color: 'rgba(0, 230, 118, 0.4)'});
    edges.push({from: 'ac_tf_3', to: 'ac_recon', color: 'rgba(124, 77, 255, 0.4)'});
    edges.push({from: 'ac_recon', to: 'ac_gl', color: 'rgba(124, 77, 255, 0.4)'});
    edges.push({from: 'ac_gl', to: 'ac_spk', color: 'rgba(124, 77, 255, 0.4)'});
    
    // Cross-connections: Audio sense feeds into acoustic pipeline
    edges.push({from: 'sens_audio', to: 'ac_mic', color: 'rgba(239, 83, 80, 0.3)'});
    
    // Self-monitoring loop: speaker output feeds back to encoder
    edges.push({from: 'ac_spk', to: 'ac_mel', color: 'rgba(255, 213, 79, 0.2)', dashes: true});
    
    // VQ tokens connect to the subconscious GRU
    edges.push({from: 'ac_vq', to: 'h_0', color: 'rgba(0, 230, 118, 0.15)'});
    edges.push({from: 'ac_vq', to: 'h_8', color: 'rgba(0, 230, 118, 0.15)'});
    edges.push({from: 'ac_vq', to: 'h_16', color: 'rgba(0, 230, 118, 0.15)'});

    nodesData = new vis.DataSet(nodes);
    edgesData = new vis.DataSet(edges);
    
    network = new vis.Network(networkContainer, {nodes: nodesData, edges: edgesData}, brainOptions);
    
    setTimeout(() => network.fit({animation: {duration: 1000, easingFunction: 'easeInOutQuad'}}), 500);
}

// Stats Elements
const conceptCount = document.getElementById('concept-count');
const expCount = document.getElementById('exp-count');
const surpriseLoss = document.getElementById('surprise-loss');

// Chemistry Bars
const chemDopamine = document.getElementById('chem-dopamine');
const chemCortisol = document.getElementById('chem-cortisol');
const chemSerotonin = document.getElementById('chem-serotonin');
const chemOxytocin = document.getElementById('chem-oxytocin');
const valDopamine = document.getElementById('val-dopamine');
const valCortisol = document.getElementById('val-cortisol');
const valSerotonin = document.getElementById('val-serotonin');
const valOxytocin = document.getElementById('val-oxytocin');

// Pre-create 128 hidden state cells
function initHiddenStateViz() {
    if (!hiddenStateContainer) return;
    for (let i = 0; i < 128; i++) {
        const cell = document.createElement('div');
        cell.className = 'hs-cell';
        hiddenStateContainer.appendChild(cell);
    }
}

// Initialize Chart.js configuration
function initCharts() {
    Chart.defaults.color = '#8a8a93';
    Chart.defaults.font.family = 'Inter';

    // 8D Emotional State Radar Chart
    const elEmotion = document.getElementById('emotionChart');
    if (elEmotion) {
        const ctxEmotion = elEmotion.getContext('2d');
        emotionChart = new Chart(ctxEmotion, {
            type: 'radar',
            data: {
                labels: ['Joy', 'Excitement', 'Trust', 'Anger', 'Surprise', 'Disgust', 'Interest', 'Love'],
                datasets: [
                    {
                        label: 'Current',
                        data: [0, 0, 0, 0, 0, 0, 0, 0],
                        backgroundColor: 'rgba(79, 195, 247, 0.2)',
                        borderColor: '#4fc3f7',
                        pointBackgroundColor: '#fff',
                        pointHoverBackgroundColor: '#fff',
                        pointHoverBorderColor: '#4fc3f7',
                        borderWidth: 2,
                    },
                    {
                        label: 'Mood Baseline',
                        data: [0, 0, 0, 0, 0, 0, 0, 0],
                        backgroundColor: 'rgba(124, 77, 255, 0.1)',
                        borderColor: 'rgba(124, 77, 255, 0.5)',
                        pointBackgroundColor: 'rgba(124, 77, 255, 0.6)',
                        borderWidth: 1,
                        borderDash: [5, 3],
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        pointLabels: { color: '#f0f0f5', font: { size: 11 } },
                        ticks: { display: false },
                        suggestedMin: -0.1,
                        suggestedMax: 0.5
                    }
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'bottom',
                        labels: { color: '#8a8a93', font: { size: 10 } }
                    }
                }
            }
        });
    }

    // 8 Maslow Drives Bar Chart
    const elDrives = document.getElementById('drivesChart');
    if (elDrives) {
        const ctxDrives = elDrives.getContext('2d');
        drivesChart = new Chart(ctxDrives, {
            type: 'bar',
            data: {
                labels: ['Sleep', 'Comfort', 'Social', 'Belonging', 'Curiosity', 'Novelty', 'Mastery', 'Autonomy'],
                datasets: [{
                    label: 'Drive Activation',
                    data: [0,0,0,0,0,0,0,0],
                    backgroundColor: [
                        '#ef5350', '#ef5350', // Survival
                        '#ffd54f', '#ffd54f', // Social (Tier 2)
                        '#4fc3f7', '#4fc3f7', '#4fc3f7', // Cognitive (Tier 3)
                        '#ba68c8'             // Self (Tier 4)
                    ],
                    borderRadius: 4
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        max: 1.0,
                        min: 0,
                        grid: { color: 'rgba(255,255,255,0.05)' }
                    },
                    y: {
                        grid: { display: false }
                    }
                },
                plugins: { legend: { display: false } }
            }
        });
    }
}

function updateUI(state) {
    if (state.status === "booting...") return;

    // Phase Status
    if (phaseBadge) phaseBadge.textContent = `Phase ${state.core.phase} (${state.core.phase_name})`;

    // Threads
    if (state.threads && threadsContainer) {
        threadsContainer.innerHTML = '';
        for (const [name, data] of Object.entries(state.threads)) {
            threadsContainer.innerHTML += `
                <div class="thread-card">
                    <span class="thread-name">${name}</span>
                    <span class="thread-ticks">${data.ticks} ticks | ${data.errors} errors</span>
                </div>
            `;
        }
    }

    // Senses
    if (state.senses && sensesContainer) {
        sensesContainer.innerHTML = `
            <div class="thread-card">
                <span class="thread-name">Vision (Camera)</span>
                <span class="thread-ticks" style="color:var(--accent)">${state.senses.vision}</span>
            </div>
            <div class="thread-card">
                <span class="thread-name">Auditory (Mic)</span>
                <span class="thread-ticks" style="color:var(--accent)">${state.senses.auditory}</span>
            </div>
            <div class="thread-card" style="grid-column: 1 / -1;">
                <span class="thread-name">Proprioception (Body)</span>
                <span class="thread-ticks">Time: ${state.senses.proprioception.time_of_day} | Fatigue: ${state.senses.proprioception.fatigue.toFixed(2)} | Uptime: ${state.senses.proprioception.uptime_hours.toFixed(1)}h</span>
            </div>
        `;
    }

    // Activity Stream
    if (state.stream && activityStream) {
        // Only autoscroll if user is already at the bottom
        let isScrolledToBottom = activityStream.scrollHeight - activityStream.clientHeight <= activityStream.scrollTop + 5;
        
        // Prevent unnecessary DOM writes if nothing changed (basic length check for efficiency)
        if (activityStream.children.length !== state.stream.length || state.stream.length > 0) {
            let html = '';
            state.stream.forEach(item => {
                const sourceClass = item.prefix === '💭' ? 'thought' : item.prefix === '👂' ? 'heard' : 'dream';
                const label = item.prefix === '💭' ? 'THOUGHT' : item.prefix === '👂' ? 'PERCEPTION' : 'SYSTEM';
                html += `
                    <div class="stream-item ${sourceClass}">
                        <div class="stream-time">
                            <span class="stream-source">${label}</span>
                            <span>${item.time}</span>
                        </div>
                        <div>${item.prefix} ${item.message}</div>
                    </div>
                `;
            });
            activityStream.innerHTML = html;
            if (isScrolledToBottom) {
                activityStream.scrollTop = activityStream.scrollHeight;
            }
        }
    }

    // Network Graph Real-Time Neural Activations update
    if (network && state.neural && state.neural.layer3_personality) {
        if (nodeCount) nodeCount.textContent = "Mega Model"; // Total nodes simulated
        
        let updates = [];
        const isSleeping = state.core.is_sleeping || false;
        const sleepPhase = state.core.current_sleep_phase || 'awake';
        
        // 1. Map Sensory State
        const inputNoise = isSleeping ? 0.1 : (Math.random() * 0.4 + 0.3);
        const inputSize = isSleeping ? 10 : (15 + (Math.random() * 5));
        if (nodesData.get('sens_vision')) {
            updates.push({id: 'sens_vision', size: inputSize, color: {background: `rgba(239, 83, 80, ${inputNoise})`}});
            updates.push({id: 'sens_audio', size: inputSize, color: {background: `rgba(239, 83, 80, ${inputNoise})`}});
            updates.push({id: 'sens_body', size: inputSize, color: {background: `rgba(239, 83, 80, ${inputNoise})`}});
        }
        
        // 2. Map Limbic State
        if (state.neurochemistry && nodesData.get('limbic_vta')) {
            const dopa = state.neurochemistry.dopamine || 0;
            const cort = state.neurochemistry.cortisol || 0;
            const pinealGlow = isSleeping ? 1.0 : 0.1;
            updates.push({id: 'limbic_vta', size: 10 + (dopa*20), color: {background: `rgba(255, 152, 0, ${0.2 + dopa})`}});
            updates.push({id: 'limbic_amygdala', size: 10 + (cort*20), color: {background: `rgba(239, 83, 80, ${0.2 + cort})`}});
            updates.push({id: 'limbic_pineal', size: 10 + (pinealGlow*10), color: {background: `rgba(186, 104, 200, ${0.2 + pinealGlow})`}});
        }
        
        // 3. Render Memory Network (Center Zone)
        if (state.network_graph && state.network_graph.nodes) {
            let memoryNodes = [];
            let memoryEdges = [];
            const totalMem = state.network_graph.nodes.length;
            
            // Generate circular layout
            state.network_graph.nodes.forEach((n, idx) => {
                const angle = (idx / totalMem) * Math.PI * 2;
                const radius = 60 + (idx * 2); 
                memoryNodes.push({
                    id: `mem_${n.id}`, 
                    label: n.label,
                    group: 'memory',
                    x: -150 + Math.cos(angle) * Math.min(radius, 150),
                    y: Math.sin(angle) * Math.min(radius, 150),
                    title: n.title
                });
            });
            
            state.network_graph.edges.forEach(e => {
                memoryEdges.push({
                    id: `e_${e.from}_${e.to}`,
                    from: `mem_${e.from}`,
                    to: `mem_${e.to}`,
                    color: 'rgba(186, 104, 200, 0.3)'
                });
            });

            // Synchronize with DataSet (Diff heavily to prevent crashing layout)
            const existingMems = nodesData.get({filter: item => item.id.toString().startsWith('mem_')});
            const existingMemEdges = edgesData.get({filter: item => item.id && item.id.toString().startsWith('e_')}); 
            
            // Only rebuild the memory cloud if the number of concepts changed
            if (existingMems.length !== memoryNodes.length) {
                nodesData.remove(existingMems.map(i => i.id));
                edgesData.remove(existingMemEdges.map(i => i.id));
                nodesData.add(memoryNodes);
                edgesData.add(memoryEdges);
            }
            
            // Pulse Memories!
            const memoryNodesCurrent = nodesData.get({filter: item => item.id.toString().startsWith('mem_')});
            if (sleepPhase === 'rem_dreaming') {
                memoryNodesCurrent.forEach(n => {
                    if (Math.random() > 0.8) {
                        updates.push({id: n.id, size: 25, color: {background: `rgba(255, 255, 255, 1.0)`}});
                    } else {
                        updates.push({id: n.id, size: 12, color: {background: `rgba(186, 104, 200, 0.4)`}});
                    }
                });
            } else {
                // Scale concept nodes by strength — strong concepts are LARGE and BRIGHT
                memoryNodesCurrent.forEach(n => {
                    // Find the original concept data for this node
                    const conceptId = n.id.replace('mem_', '');
                    const conceptNode = state.network_graph.nodes.find(c => c.id === conceptId);
                    const strength = conceptNode ? (conceptNode.strength || 0.1) : 0.1;
                    const sz = 8 + (strength * 25); // 8px at 0 strength, 33px at full
                    const alpha = 0.3 + (strength * 0.7); // 0.3 at 0, 1.0 at full
                    updates.push({id: n.id, size: sz, color: {background: `rgba(186, 104, 200, ${alpha})`}});
                });
            }
        }

        // 4. Update 128-dim Personality Core Hidden State — VIVID diverging colormap
        const hs = state.neural.layer3_personality.hidden_state_activation || [];
        for (let i = 0; i < hs.length && i < 128; i++) {
            const val = hs[i];
            // Red for negative, green for positive, bright always
            let r, g, b;
            if (val >= 0) {
                // Positive: black → bright green
                const t = Math.min(val * 2.5, 1.0);
                r = Math.floor(30 * (1 - t));
                g = Math.floor(80 + 175 * t);
                b = Math.floor(30 * (1 - t));
            } else {
                // Negative: black → bright red
                const t = Math.min(Math.abs(val) * 2.5, 1.0);
                r = Math.floor(80 + 175 * t);
                g = Math.floor(30 * (1 - t));
                b = Math.floor(30 * (1 - t));
            }
            const brightness = Math.max(0.4, Math.min(Math.abs(val) * 3, 1.0));
            updates.push({
                id: `h_${i}`,
                color: { background: `rgba(${r}, ${g}, ${b}, ${brightness})` },
                size: 7 + (Math.abs(val) * 10)
            });
        }
        
        // 5. Output node flashes based on World Model prediction loss 
        const loss = state.neural.layer4_world_model.last_loss || 0;
        for (let i=0; i<8; i++) {
            const outIntensity = Math.min(0.2 + (loss * 10), 0.9) + (Math.random()*0.2);
            updates.push({id: `out_${i}`, size: 10 + (Math.random() * 4), color: {background: `rgba(255, 213, 79, ${outIntensity})`}});
        }

        // 6. V7: Acoustic Pipeline Node Animation
        if (state.acoustic_pipeline) {
            const ap = state.acoustic_pipeline;
            const hasActivity = (ap.total_interactions || 0) > 0;
            const baseGlow = hasActivity ? 0.6 : 0.2;
            const pulse = Math.sin(Date.now() / 500) * 0.2 + 0.5;
            
            // Mic & Speaker pulse
            if (nodesData.get('ac_mic')) {
                updates.push({id: 'ac_mic', size: 15 + pulse * 8, color: {background: `rgba(239, 83, 80, ${0.3 + pulse * 0.3})`}});
                updates.push({id: 'ac_spk', size: 15 + pulse * 5, color: {background: `rgba(239, 83, 80, ${0.3 + pulse * 0.2})`}});
            }
            
            // Auditory Cortex glow based on frames
            const acFrames = (ap.auditory_cortex?.frames_processed || 0);
            const acGlow = Math.min(0.3 + (acFrames / 200), 1.0);
            if (nodesData.get('ac_mel')) {
                updates.push({id: 'ac_mel', size: 14 + acGlow * 8, color: {background: `rgba(0, 230, 118, ${acGlow})`}});
                updates.push({id: 'ac_enc', size: 14 + acGlow * 6, color: {background: `rgba(0, 230, 118, ${acGlow * 0.9})`}});
            }
            
            // VQ Codebook: intensity based on utilization
            const vqUtil = ap.vq_codebook?.codebook_utilization || 0;
            if (nodesData.get('ac_vq')) {
                updates.push({id: 'ac_vq', size: 16 + vqUtil * 30, color: {background: `rgba(0, 230, 118, ${0.3 + vqUtil})`}});
            }
            
            // Transformer layers: sequential pulse
            for (let i = 0; i < 4; i++) {
                const layerPulse = Math.sin((Date.now() / 400) + (i * 0.8)) * 0.3 + 0.5;
                const seqs = ap.acoustic_brain?.total_sequences_heard || 0;
                const tfGlow = Math.min(0.2 + (seqs / 50) + layerPulse * 0.3, 1.0);
                if (nodesData.get(`ac_tf_${i}`)) {
                    updates.push({id: `ac_tf_${i}`, size: 14 + tfGlow * 8, color: {background: `rgba(0, 230, 118, ${tfGlow})`}});
                }
            }
            
            // Vocoder: glow based on syntheses
            const synths = ap.vocoder?.total_syntheses || 0;
            const vocGlow = Math.min(0.2 + (synths / 20), 1.0);
            if (nodesData.get('ac_recon')) {
                updates.push({id: 'ac_recon', size: 12 + vocGlow * 8, color: {background: `rgba(124, 77, 255, ${vocGlow})`}});
                updates.push({id: 'ac_gl', size: 12 + vocGlow * 6, color: {background: `rgba(124, 77, 255, ${vocGlow * 0.8})`}});
            }
        }
        
        if (nodesData.get('h_0')) {
             nodesData.update(updates);
        }
    }

    // Working Memory
    if (state.working_memory && wmUsage && wmCapacity && wmContainer) {
        wmUsage.textContent = state.working_memory.usage;
        wmCapacity.textContent = state.working_memory.capacity;
        wmContainer.innerHTML = '';
        
        if (state.working_memory.slots.length === 0) {
            wmContainer.innerHTML = '<div style="color:var(--text-muted); font-size:0.8rem; padding:1rem; text-align:center;">Buffer is empty</div>';
        } else {
            state.working_memory.slots.forEach(slot => {
                wmContainer.innerHTML += `
                    <div class="wm-slot">
                        <span class="wm-concept">${slot.concept}</span>
                        <div class="wm-meta">
                            <span>${slot.source}</span>
                            <span class="wm-salience">${slot.salience.toFixed(2)}</span>
                        </div>
                    </div>
                `;
            });
        }
    }

    // Neurochemistry
    if (state.neurochemistry && chemDopamine && chemCortisol && chemSerotonin && chemOxytocin) {
        const d = state.neurochemistry.dopamine || 0;
        const c = state.neurochemistry.cortisol || 0;
        const s = state.neurochemistry.serotonin || 0;
        const o = state.neurochemistry.oxytocin || 0;
        
        chemDopamine.style.width = `${d * 100}%`;
        chemCortisol.style.width = `${c * 100}%`;
        chemSerotonin.style.width = `${s * 100}%`;
        chemOxytocin.style.width = `${o * 100}%`;
        
        if (valDopamine) valDopamine.textContent = d.toFixed(3);
        if (valCortisol) valCortisol.textContent = c.toFixed(3);
        if (valSerotonin) valSerotonin.textContent = s.toFixed(3);
        if (valOxytocin) valOxytocin.textContent = o.toFixed(3);
    }

    // Neural Stats
    if (state.neural) {
        if (conceptCount) conceptCount.textContent = state.neural.layer2_binding.learned_concepts.toLocaleString();
        if (expCount) expCount.textContent = state.neural.layer3_personality.total_experiences.toLocaleString();
        if (surpriseLoss) surpriseLoss.textContent = state.neural.layer4_world_model.last_loss.toFixed(4);
        
        const totalParamsEl = document.getElementById('total-params');
        if (totalParamsEl && state.neural.total_parameters) {
            totalParamsEl.textContent = state.neural.total_parameters.toLocaleString() + " Params";
        }

        // Update hidden state cells
        if (hiddenStateContainer) {
            const cells = hiddenStateContainer.children;
            const hs = state.neural.layer3_personality.hidden_state_activation || [];
            for (let i = 0; i < cells.length; i++) {
                if (i < hs.length) {
                    // Map activation value to opacity/color
                    const val = hs[i];
                    // normalize roughly from -1 to 1 for visual
                    const intensity = Math.min(Math.max((val + 1) / 2, 0), 1);
                    cells[i].style.backgroundColor = `rgba(79, 195, 247, ${intensity})`;
                } else {
                    cells[i].style.backgroundColor = 'rgba(255,255,255,0.05)';
                }
            }
        }
    }

    // Emotions Radar Chart Update (with mood baseline)
    if (state.emotions && emotionChart) {
        emotionChart.data.datasets[0].data = state.emotions;
        // Update mood baseline if available
        if (state.mood_baseline) {
            emotionChart.data.datasets[1].data = state.mood_baseline;
        }
        emotionChart.update('none');
    }

    // Drives Bar Chart Update
    if (state.drives && drivesChart) {
        drivesChart.data.datasets[0].data = [
            state.drives.sleep?.level || 0,
            state.drives.comfort?.level || 0,
            state.drives.social?.level || 0,
            state.drives.belonging?.level || 0,
            state.drives.curiosity?.level || 0,
            state.drives.novelty?.level || 0,
            state.drives.mastery?.level || 0,
            state.drives.autonomy?.level || 0,
        ];
        drivesChart.update('none');
    }

    // Language Acquisition (V6 -> V7 Acoustic Memory)
    if (state.language_acquisition) {
        const la = state.language_acquisition;
        const modeTag = document.getElementById('grammar-mode-tag');
        if (modeTag) modeTag.textContent = la.grammar_mode || 'acoustic_v7';

        // Vocabulary stats (mapped to V7 Acoustic Memory)
        const vocabSize = document.getElementById('la-vocab-size');
        const wordsHeard = document.getElementById('la-words-heard');
        const sentencesHeard = document.getElementById('la-sentences-heard');
        
        if (vocabSize) {
            vocabSize.textContent = la.acoustic_memory?.words || la.joint_attention?.vocabulary_size || 0;
        }
        if (wordsHeard) {
            wordsHeard.textContent = (la.acoustic_memory?.total_recognitions || la.ngram?.total_words_heard || 0).toLocaleString();
        }
        if (sentencesHeard) {
            sentencesHeard.textContent = la.ngram?.total_sentences_heard || 0; // V7 does not track sentences yet
        }

        // Babbling stats
        const repertoire = document.getElementById('la-repertoire');
        const babbles = document.getElementById('la-babbles');
        const reinforcements = document.getElementById('la-reinforcements');
        const lastBabble = document.getElementById('la-last-babble');
        
        // Use v6 babbling if it exists, otherwise pull from V7 vocoder
        if (repertoire) repertoire.textContent = la.babbling?.repertoire_size || 0;
        if (babbles) babbles.textContent = state.acoustic_pipeline?.vocoder?.total_syntheses || la.babbling?.total_babbles || 0;
        if (reinforcements) reinforcements.textContent = la.babbling?.total_reinforcements || 0;
        if (lastBabble) lastBabble.textContent = la.babbling?.last_babble || '---';

        // Cross-modal bindings
        const bindingsEl = document.getElementById('la-bindings');
        const learnedEl = document.getElementById('la-learned');
        const permanentEl = document.getElementById('la-permanent');
        if (bindingsEl && la.joint_attention) bindingsEl.textContent = la.joint_attention.total_bindings || 0;
        if (learnedEl && la.joint_attention) learnedEl.textContent = la.joint_attention.learned_bindings || 0;
        if (permanentEl && la.joint_attention) permanentEl.textContent = la.joint_attention.permanent_bindings || 0;

        // Strongest bindings list
        const container = document.getElementById('la-strongest-container');
        if (container && la.joint_attention && la.joint_attention.strongest_bindings) {
            const bindings = la.joint_attention.strongest_bindings;
            if (bindings.length > 0) {
                let html = '<div style="font-size:0.7rem; color:var(--text-muted); margin-bottom:0.3rem;">Strongest Bindings:</div>';
                bindings.forEach(b => {
                    const barWidth = (b.strength * 100).toFixed(0);
                    html += `<div style="display:flex; align-items:center; gap:0.5rem; margin-bottom:0.2rem; font-size:0.75rem;">
                        <span style="min-width:60px; color:var(--accent-primary);">${b.visual}</span>
                        <span style="color:var(--text-muted);">&harr;</span>
                        <span style="min-width:60px; color:var(--accent-secondary);">${b.word}</span>
                        <div style="flex:1; height:4px; background:rgba(255,255,255,0.05); border-radius:2px; overflow:hidden;">
                            <div style="width:${barWidth}%; height:100%; background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary)); border-radius:2px;"></div>
                        </div>
                        <span class="mono" style="font-size:0.65rem;">${b.strength.toFixed(2)}</span>
                    </div>`;
                });
                container.innerHTML = html;
            }
        }
    }

    // Acoustic Neural Pipeline (V7) — Enhanced
    if (state.acoustic_pipeline) {
        const ap = state.acoustic_pipeline;

        // Overview stats
        const paramsTag = document.getElementById('acoustic-params');
        if (paramsTag) paramsTag.textContent = (ap.total_params || 0).toLocaleString() + ' Params';
        const apTotal = document.getElementById('ap-total-params');
        if (apTotal) apTotal.textContent = (ap.total_params || 0).toLocaleString();
        const apInter = document.getElementById('ap-interactions');
        if (apInter) apInter.textContent = ap.total_interactions || 0;
        const apCtx = document.getElementById('ap-context');
        if (apCtx) apCtx.textContent = ap.context_buffer_size || 0;

        // Auditory Cortex
        if (ap.auditory_cortex) {
            const ac = ap.auditory_cortex;
            const acTag = document.getElementById('ac-params-tag');
            if (acTag) acTag.textContent = (ac.params || 0).toLocaleString() + ' params';
            const acFrames = document.getElementById('ac-frames');
            if (acFrames) acFrames.textContent = (ac.frames_processed || 0).toLocaleString();
            const acLoss = document.getElementById('ac-loss');
            if (acLoss) acLoss.textContent = (ac.avg_loss || 0).toFixed(4);
        }

        // VQ Codebook
        if (ap.vq_codebook) {
            const vq = ap.vq_codebook;
            const vqTag = document.getElementById('vq-tag');
            if (vqTag) vqTag.textContent = (vq.active_codes || 0) + ' / 256 active';
            const vqActive = document.getElementById('vq-active');
            if (vqActive) vqActive.textContent = vq.active_codes || 0;
            const vqUtil = document.getElementById('vq-util');
            if (vqUtil) vqUtil.textContent = ((vq.codebook_utilization || 0) * 100).toFixed(1) + '%';
            const vqQuant = document.getElementById('vq-quant');
            if (vqQuant) vqQuant.textContent = (vq.total_quantizations || 0).toLocaleString();

            // Codebook grid visualization
            const grid = document.getElementById('codebook-viz');
            if (grid && grid.children.length === 0) {
                for (let i = 0; i < 256; i++) {
                    const cell = document.createElement('div');
                    cell.className = 'codebook-cell';
                    cell.id = 'cb-' + i;
                    cell.title = 'Token ' + i;
                    grid.appendChild(cell);
                }
            }
            // Mark active codes
            if (grid && vq.active_codes > 0) {
                const activeCount = vq.active_codes;
                for (let i = 0; i < 256; i++) {
                    const cell = document.getElementById('cb-' + i);
                    if (cell) cell.className = i < activeCount ? 'codebook-cell active' : 'codebook-cell';
                }
            }
        }

        // Acoustic Language Model
        if (ap.acoustic_brain) {
            const ab = ap.acoustic_brain;
            const almTag = document.getElementById('alm-params-tag');
            if (almTag) almTag.textContent = (ab.params || 0).toLocaleString() + ' params';
            const almSeqs = document.getElementById('alm-seqs');
            if (almSeqs) almSeqs.textContent = ab.total_sequences_heard || 0;
            const almTokens = document.getElementById('alm-tokens');
            if (almTokens) almTokens.textContent = (ab.total_tokens_seen || 0).toLocaleString();
            const almLoss = document.getElementById('alm-loss');
            if (almLoss) almLoss.textContent = (ab.avg_loss || 0).toFixed(4);
            // Loss bar (scale: 0-6 = full, closer to 0 = better)
            const lossBar = document.getElementById('alm-loss-bar');
            if (lossBar) {
                const pct = Math.min(100, ((ab.avg_loss || 6) / 6) * 100);
                lossBar.style.width = pct + '%';
            }
        }

        // Neural Vocoder
        if (ap.vocoder) {
            const vo = ap.vocoder;
            const nvTag = document.getElementById('nv-params-tag');
            if (nvTag) nvTag.textContent = (vo.params || 0).toLocaleString() + ' params';
            const nvSynths = document.getElementById('nv-synths');
            if (nvSynths) nvSynths.textContent = vo.total_syntheses || 0;
            const nvSr = document.getElementById('nv-sr');
            if (nvSr) nvSr.textContent = (vo.sample_rate || 16000) / 1000 + 'kHz';
        }
    }
}
// Fetch loop
async function fetchState() {
    try {
        const response = await fetch('/api/state');
        if (response.ok) {
            const data = await response.json();
            updateUI(data);
        }
    } catch (e) {
        console.error("Dashboard disconnected", e);
    }
}

// Init
window.addEventListener('DOMContentLoaded', () => {
    initHiddenStateViz();
    initCharts();
    initNetwork();
    
    // Poll every 1 second
    setInterval(fetchState, 1000);
    fetchState(); // initial fetch
});
