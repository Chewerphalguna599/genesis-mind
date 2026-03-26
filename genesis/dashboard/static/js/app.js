// Genesis Mind V5 — Dashboard Application Logic

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

const networkOptions = {
    nodes: {
        shape: 'dot',
        size: 20,
        font: { size: 14, color: '#f0f0f5', face: 'Inter', strokeWidth: 3, strokeColor: '#0a0a0c' },
        borderWidth: 2,
        color: {
            background: 'rgba(79, 195, 247, 0.2)',
            border: '#4fc3f7',
            highlight: { background: '#4fc3f7', border: '#fff' }
        }
    },
    edges: {
        width: 1.5,
        color: { color: 'rgba(255,255,255,0.15)', highlight: '#4fc3f7' },
        smooth: { type: 'continuous' }
    },
    physics: {
        barnesHut: { gravitationalConstant: -4000, centralGravity: 0.1, springLength: 150, springConstant: 0.02 }
    }
};

function initNetwork() {
    if (networkContainer && !network) {
        network = new vis.Network(networkContainer, {nodes: nodesData, edges: edgesData}, networkOptions);
    }
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
                labels: ['Joy', 'Fear', 'Anger', 'Sadness', 'Trust', 'Disgust', 'Anticipation', 'Surprise'],
                datasets: [{
                    label: 'Current Emotional Vector',
                    data: [0, 0, 0, 0, 0, 0, 0, 0],
                    backgroundColor: 'rgba(79, 195, 247, 0.2)',
                    borderColor: '#4fc3f7',
                    pointBackgroundColor: '#fff',
                    pointHoverBackgroundColor: '#fff',
                    pointHoverBorderColor: '#4fc3f7',
                    borderWidth: 2,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    r: {
                        angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        pointLabels: { color: '#f0f0f5', font: { size: 11 } },
                        ticks: { display: false, max: 1.0, min: -0.2 }
                    }
                },
                plugins: { legend: { display: false } }
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
                labels: ['Fatigue', 'Hunger', 'Fear', 'Novelty', 'Curiosity', 'Social', 'Play', 'Meaning'],
                datasets: [{
                    label: 'Drive Activation',
                    data: [0,0,0,0,0,0,0,0],
                    backgroundColor: [
                        '#ef5350', '#ef5350', // Survival
                        '#ffd54f', '#ffd54f', // Security 
                        '#4fc3f7', '#4fc3f7', '#4fc3f7', // Social/Cognitive
                        '#ba68c8'             // Transcendence
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

    // Network Graph
    if (state.network_graph) {
        if (nodeCount) nodeCount.textContent = state.network_graph.nodes.length;
        
        // Update nodes and edges dynamically
        nodesData.update(state.network_graph.nodes);
        
        // Edges need a unique ID to be updated cleanly without duplicating
        const formattedEdges = state.network_graph.edges.map(e => ({
            id: `${e.from}-${e.to}`,
            from: e.from,
            to: e.to
        }));
        edgesData.update(formattedEdges);
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

    // Emotions Radar Chart Update
    if (state.emotions && emotionChart) {
        emotionChart.data.datasets[0].data = state.emotions;
        emotionChart.update('none'); // Update without full animation for performance
    }

    // Drives Bar Chart Update
    if (state.drives && drivesChart) {
        drivesChart.data.datasets[0].data = [
            state.drives.fatigue?.level || 0,
            state.drives.hunger?.level || 0,
            state.drives.fear?.level || 0,
            state.drives.novelty?.level || 0,
            state.drives.curiosity?.level || 0,
            state.drives.social?.level || 0,
            state.drives.play?.level || 0,
            state.drives.meaning?.level || 0,
        ];
        drivesChart.update('none');
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
