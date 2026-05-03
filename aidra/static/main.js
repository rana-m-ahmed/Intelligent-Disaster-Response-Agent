const socket = io();

let scene, camera, renderer, controls;
let gridMeshes = {};
let ambulanceMeshes = {};
let victimMeshes = {};
let rescuedVictimMeshes = {};
let medCenterMeshes = [];
let routeLines = {};
let routeDrawAnimations = {};
let pendingRouteAnimations = {};
let fadingRouteLines = [];
let gameState = {};
let animationId;
let stepIntervalId = null;
let blockModeActive = false;
let hoveredCellKey = null;

const routeColors = [0x00ff99, 0x00ccff, 0xff99ff, 0xffaa00];

function init() {
    const container = document.getElementById("canvas-container");
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0e27);

    camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    camera.position.set(15, 20, 15);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.shadowMap.enabled = true;
    container.appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(5, 0, 5);
    controls.distance = 25;

    const ambientLight = new THREE.AmbientLight(0x00ff99, 0.3);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0x00ffff, 0.6);
    directionalLight.position.set(10, 20, 10);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    const groundGeo = new THREE.PlaneGeometry(10, 10);
    const groundMat = new THREE.MeshStandardMaterial({ color: 0x1a1f3a, emissive: 0x003366 });
    const ground = new THREE.Mesh(groundGeo, groundMat);
    ground.rotation.x = -Math.PI / 2;
    ground.receiveShadow = true;
    scene.add(ground);

    window.addEventListener("resize", onWindowResize);
    renderer.domElement.addEventListener("click", onCanvasClick);
    renderer.domElement.addEventListener("mousemove", onCanvasMouseMove);

    document.getElementById("btn-scenario-a").addEventListener("click", () => initScenario("A", "btn-scenario-a"));
    document.getElementById("btn-scenario-b").addEventListener("click", () => initScenario("B", "btn-scenario-b"));
    document.getElementById("btn-scenario-c").addEventListener("click", () => initScenario("C", "btn-scenario-c"));

    document.getElementById("alpha-slider").addEventListener("input", (e) => {
        document.getElementById("alpha-value").textContent = parseFloat(e.target.value).toFixed(1);
        updateTradeoffBanner();
    });

    document.getElementById("btn-pause").addEventListener("click", () => {
        socket.emit("pause");
        document.getElementById("btn-pause").classList.add("btn-active");
        document.getElementById("btn-resume").classList.remove("btn-active");
    });

    document.getElementById("btn-resume").addEventListener("click", () => {
        socket.emit("resume");
        document.getElementById("btn-resume").classList.add("btn-active");
        document.getElementById("btn-pause").classList.remove("btn-active");
    });

    document.getElementById("btn-reset").addEventListener("click", () => {
        location.reload();
    });

    document.getElementById("btn-plan-first").addEventListener("click", () => {
        const first = (gameState.victims || []).find((v) => !v.rescued);
        if (!first) {
            appendLogEntry("USER ACTION", "No active victims available to plan.", "ASSIGNMENT");
            return;
        }
        emitPlanRoute(first.id);
    });

    document.getElementById("btn-plan-full").addEventListener("click", () => {
        const algorithm = document.getElementById("algorithm-select").value;
        const alpha = parseFloat(document.getElementById("alpha-slider").value);
        socket.emit("plan_full_rescue", { algorithm, alpha });
        appendLogEntry("USER ACTION", `Planning full rescue with ${algorithm.toUpperCase()} (alpha=${alpha.toFixed(1)}).`, "ASSIGNMENT");
    });

    document.getElementById("btn-block-mode").addEventListener("click", () => {
        blockModeActive = !blockModeActive;
        const btn = document.getElementById("btn-block-mode");
        btn.textContent = `🚧 Block Mode: ${blockModeActive ? "ON" : "OFF"}`;
        btn.style.background = blockModeActive ? "linear-gradient(135deg, #ff6600, #ff3300)" : "";
        renderer.domElement.style.cursor = blockModeActive ? "crosshair" : "default";
    });

    initScenario("A", "btn-scenario-a");
    startStepInterval(500);
    animate();
}

function initScenario(scenario, buttonId) {
    resetClientScenarioView();
    setActiveScenarioButton(buttonId);
    socket.emit("init_scenario", { scenario });
}

function resetClientScenarioView() {
    clearRouteLines();
    Object.values(victimMeshes).forEach((mesh) => scene.remove(mesh));
    Object.values(rescuedVictimMeshes).forEach((mesh) => scene.remove(mesh));
    Object.values(ambulanceMeshes).forEach((mesh) => scene.remove(mesh));
    victimMeshes = {};
    rescuedVictimMeshes = {};
    ambulanceMeshes = {};
    routeDrawAnimations = {};
    pendingRouteAnimations = {};
    fadingRouteLines = [];
    const logContainer = document.getElementById("log-container");
    logContainer.innerHTML = '<p class="log-placeholder">Waiting for events...</p>';
}

function setActiveScenarioButton(buttonId) {
    ["btn-scenario-a", "btn-scenario-b", "btn-scenario-c"].forEach((id) => {
        document.getElementById(id).classList.remove("btn-scenario-active");
    });
    document.getElementById(buttonId).classList.add("btn-scenario-active");
}

function startStepInterval(ms = 500) {
    if (stepIntervalId) {
        return;
    }
    stepIntervalId = setInterval(() => {
        socket.emit("step_simulation");
    }, ms);
}

function getTimeStamp() {
    return new Date().toLocaleTimeString("en-US", { hour12: false });
}

function appendLogEntry(type, message, cssClass = "") {
    const logContainer = document.getElementById("log-container");
    const placeholder = logContainer.querySelector(".log-placeholder");
    if (placeholder) {
        placeholder.remove();
    }
    const entry = document.createElement("div");
    entry.className = `log-entry ${cssClass}`.trim();
    entry.innerHTML = `
        <span class="log-timestamp">${getTimeStamp()}</span><br>
        <span class="log-type">${type}</span><br>
        <span class="log-justification">${message}</span>
    `;
    logContainer.appendChild(entry);
    while (logContainer.children.length > 50) {
        logContainer.removeChild(logContainer.firstChild);
    }
    logContainer.scrollTop = logContainer.scrollHeight;
}

function emitPlanRoute(victimId) {
    const algorithm = document.getElementById("algorithm-select").value;
    const alpha = parseFloat(document.getElementById("alpha-slider").value);
    socket.emit("plan_route", { victim_id: victimId, algorithm, alpha });
    appendLogEntry("USER ACTION", `Clicked ${victimId} - planning route with ${algorithm.toUpperCase()} (alpha=${alpha.toFixed(1)}).`, "ROUTE_SELECTION");
}

function onCanvasMouseMove(event) {
    const hit = raycastGrid(event);
    if (!hit) {
        if (hoveredCellKey && gridMeshes[hoveredCellKey]) {
            gridMeshes[hoveredCellKey].material.emissiveIntensity = 0.3;
        }
        hoveredCellKey = null;
        renderer.domElement.style.cursor = blockModeActive ? "crosshair" : "default";
        return;
    }

    const key = `${hit.userData.coords[0]},${hit.userData.coords[1]}`;
    if (hoveredCellKey && hoveredCellKey !== key && gridMeshes[hoveredCellKey]) {
        gridMeshes[hoveredCellKey].material.emissiveIntensity = 0.3;
    }
    hoveredCellKey = key;
    hit.material.emissiveIntensity = 0.7;
    renderer.domElement.style.cursor = blockModeActive ? "crosshair" : "default";
}

function onCanvasClick(event) {
    const victimHit = raycastVictims(event);
    if (victimHit && !event.shiftKey && !event.ctrlKey) {
        emitPlanRoute(victimHit.userData.victimId);
        return;
    }

    const gridHit = raycastGrid(event);
    if (!gridHit || !gridHit.userData || !gridHit.userData.coords) {
        return;
    }
    const coords = gridHit.userData.coords;
    const cellType = gridHit.userData.cellType;

    if (event.shiftKey) {
        markCellBlocked(coords);
        socket.emit("trigger_event", { type: "block", coords });
        return;
    }
    if (event.ctrlKey) {
        socket.emit("trigger_event", { type: "new_victim", coords });
        return;
    }
    if (blockModeActive && cellType !== "BLOCKED") {
        markCellBlocked(coords);
        socket.emit("trigger_event", { type: "block", coords });
    }
}

function markCellBlocked(coords) {
    const key = `${coords[0]},${coords[1]}`;
    const mesh = gridMeshes[key];
    if (!mesh || mesh.userData.cellType === "BLOCKED") {
        return;
    }
    mesh.userData.cellType = "BLOCKED";
    mesh.material.color.setHex(0x1f0a0a);
    mesh.material.emissive.setHex(0xff0000);
}

function onWindowResize() {
    const w = document.getElementById("canvas-container").clientWidth;
    const h = document.getElementById("canvas-container").clientHeight;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h);
}

function animate() {
    animationId = requestAnimationFrame(animate);
    const now = Date.now();

    Object.values(victimMeshes).forEach((mesh) => {
        const offset = mesh.userData.pulseOffset || 0;
        const scale = 1 + 0.15 * Math.sin(now * 0.003 + offset);
        mesh.scale.setScalar(scale);
        mesh.position.y = 0.5 + 0.05 * Math.sin(now * 0.004 + offset);
    });

    Object.values(ambulanceMeshes).forEach((group) => {
        const currentPos = group.userData.currentPos;
        const targetPos = group.userData.targetPos;
        if (currentPos && targetPos) {
            currentPos.lerp(targetPos, 0.12);
            group.position.set(currentPos.x, 0.2, currentPos.z);
        }
        const beacon = group.children[1];
        if (beacon && beacon.material) {
            beacon.material.emissiveIntensity = 0.2 + 0.8 * Math.abs(Math.sin(now * 0.005));
        }
    });

    Object.keys(routeDrawAnimations).forEach((ambId) => {
        const anim = routeDrawAnimations[ambId];
        anim.drawRange += 1;
        anim.line.geometry.setDrawRange(0, Math.min(anim.drawRange, anim.pointsCount));
        if (anim.drawRange >= anim.pointsCount) {
            delete routeDrawAnimations[ambId];
        }
    });

    fadingRouteLines = fadingRouteLines.filter((line) => {
        line.material.opacity -= 0.03;
        if (line.material.opacity <= 0) {
            scene.remove(line);
            return false;
        }
        return true;
    });

    controls.update();
    renderer.render(scene, camera);
}

function raycastFromMouse(event, objects) {
    const rect = renderer.domElement.getBoundingClientRect();
    const mouse = new THREE.Vector2();
    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(mouse, camera);
    const intersections = raycaster.intersectObjects(objects, true);
    return intersections.length > 0 ? intersections[0].object : null;
}

function raycastVictims(event) {
    return raycastFromMouse(event, Object.values(victimMeshes));
}

function raycastGrid(event) {
    return raycastFromMouse(event, Object.values(gridMeshes));
}

function createGridCell(x, y, cellType) {
    const size = 0.9;
    const colorMap = {
        SAFE: 0x0a3a1f,
        RISK: 0x5a1f0f,
        BLOCKED: 0x1f0a0a,
        MED_CENTER: 0x0a1f5a,
        VICTIM: 0x5a5a0a,
    };
    const emissiveMap = {
        SAFE: 0x00ff99,
        RISK: 0xff6600,
        BLOCKED: 0xff0000,
        MED_CENTER: 0x00ffff,
        VICTIM: 0xffff00,
    };
    const geometry = new THREE.BoxGeometry(size, 0.1, size);
    const material = new THREE.MeshStandardMaterial({
        color: colorMap[cellType] || 0x0a3a1f,
        emissive: emissiveMap[cellType] || 0x000000,
        emissiveIntensity: 0.3,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(x, 0, y);
    mesh.userData = { coords: [x, y], cellType };
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    return mesh;
}

function createAmbulance() {
    const group = new THREE.Group();
    const body = new THREE.Mesh(
        new THREE.BoxGeometry(0.3, 0.2, 0.3),
        new THREE.MeshStandardMaterial({ color: 0x00ff99 })
    );
    group.add(body);

    const beacon = new THREE.Mesh(
        new THREE.CylinderGeometry(0.15, 0.15, 0.05),
        new THREE.MeshStandardMaterial({ color: 0xff0000, emissive: 0xff0000, emissiveIntensity: 0.5 })
    );
    beacon.position.y = 0.2;
    beacon.castShadow = true;
    group.add(beacon);
    return group;
}

function createVictim(severity) {
    const severityColor = { critical: 0xff0000, moderate: 0xffaa00, minor: 0xffff00 };
    const color = severityColor[severity] || 0xffff00;
    const mesh = new THREE.Mesh(
        new THREE.IcosahedronGeometry(0.15, 3),
        new THREE.MeshStandardMaterial({ color, emissive: color, emissiveIntensity: 0.4 })
    );
    mesh.castShadow = true;
    return mesh;
}

function createRescuedMarker() {
    return new THREE.Mesh(
        new THREE.CylinderGeometry(0.2, 0.2, 0.02, 24),
        new THREE.MeshStandardMaterial({ color: 0x00ff99, emissive: 0x00ff99, emissiveIntensity: 0.9 })
    );
}

function createMedCenter() {
    const group = new THREE.Group();
    const material = new THREE.MeshStandardMaterial({ color: 0x00ffff, emissive: 0x00ffff, emissiveIntensity: 0.5 });
    const cross1 = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.05, 0.05), material);
    const cross2 = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.05, 0.4), material);
    group.add(cross1);
    group.add(cross2);
    return group;
}

function clearRouteLines() {
    Object.keys(routeLines).forEach((ambId) => {
        scene.remove(routeLines[ambId]);
    });
    routeLines = {};
}

function updateTradeoffBanner() {
    const alpha = parseFloat(document.getElementById("alpha-slider").value);
    const text = alpha < 3 ? "⚡ Speed Priority" : alpha > 7 ? "🛡 Safety Priority" : "⚖ Balanced Approach";
    document.getElementById("tradeoff-banner").textContent = text;
}

function formatAssignmentPlan(plan) {
    if (!plan || Object.keys(plan).length === 0) {
        return "None active";
    }
    return Object.entries(plan)
        .map(([amb, victim]) => `${amb} -> ${victim}`)
        .join(" | ");
}

function formatSurvivalList(victims) {
    if (!victims || victims.length === 0) {
        return "None";
    }
    return victims
        .map((victim) => `${victim.id}:${Number(victim.survival_prob ?? 0).toFixed(2)}`)
        .join(" | ");
}

socket.on("state_update", (data) => {
    gameState = data;

    // Full route cleanup before redraw.
    clearRouteLines();

    Object.values(gridMeshes).forEach((mesh) => scene.remove(mesh));
    gridMeshes = {};
    for (let x = 0; x < 10; x++) {
        for (let y = 0; y < 10; y++) {
            const cellType = data.grid[x] ? data.grid[x][y] : "SAFE";
            const mesh = createGridCell(x, y, cellType);
            scene.add(mesh);
            gridMeshes[`${x},${y}`] = mesh;
        }
    }

    Object.values(victimMeshes).forEach((mesh) => scene.remove(mesh));
    Object.values(rescuedVictimMeshes).forEach((mesh) => scene.remove(mesh));
    victimMeshes = {};
    rescuedVictimMeshes = {};
    data.victims.forEach((victim, index) => {
        const [x, y] = victim.pos;
        if (victim.rescued) {
            const rescuedMarker = createRescuedMarker();
            rescuedMarker.position.set(x, 0.06, y);
            scene.add(rescuedMarker);
            rescuedVictimMeshes[victim.id] = rescuedMarker;
            return;
        }

        const victimMesh = createVictim(victim.severity);
        victimMesh.position.set(x, 0.5, y);
        if (victimMesh.material) {
            victimMesh.material.transparent = true;
            victimMesh.material.opacity = 0.35 + 0.65 * Math.max(0, Math.min(1, victim.survival_prob ?? 0.5));
        }
        victimMesh.userData = {
            victimId: victim.id,
            survival: victim.survival_prob,
            pulseOffset: index * 1.2,
        };
        scene.add(victimMesh);
        victimMeshes[victim.id] = victimMesh;
    });

    Object.values(ambulanceMeshes).forEach((mesh) => {
        mesh.userData.stale = true;
    });
    data.ambulances.forEach((amb, ambIndex) => {
        const target = new THREE.Vector3(amb.pos[0], 0.2, amb.pos[1]);
        if (!ambulanceMeshes[amb.id]) {
            const group = createAmbulance();
            group.position.set(target.x, target.y, target.z);
            group.userData = {
                ambulanceId: amb.id,
                currentPos: target.clone(),
                targetPos: target.clone(),
                stale: false,
            };
            scene.add(group);
            ambulanceMeshes[amb.id] = group;
        } else {
            ambulanceMeshes[amb.id].userData.targetPos = target;
            ambulanceMeshes[amb.id].userData.stale = false;
        }

        if (amb.route && amb.route.length > 1) {
            const points = amb.route.map((p) => new THREE.Vector3(p[0], 0.1, p[1]));
            const geometry = new THREE.BufferGeometry().setFromPoints(points);
            const color = routeColors[ambIndex % routeColors.length];
            const material = new THREE.LineBasicMaterial({ color, linewidth: 2, transparent: true, opacity: 0.95 });
            const line = new THREE.Line(geometry, material);
            const pending = pendingRouteAnimations[amb.id];
            if (pending && pending.pointsCount === points.length) {
                geometry.setDrawRange(0, Math.min(2, points.length));
                routeDrawAnimations[amb.id] = { line, drawRange: 2, pointsCount: points.length };
                delete pendingRouteAnimations[amb.id];
            } else {
                geometry.setDrawRange(0, points.length);
            }
            scene.add(line);
            routeLines[amb.id] = line;
        }
    });

    Object.keys(ambulanceMeshes).forEach((ambId) => {
        if (ambulanceMeshes[ambId].userData.stale) {
            scene.remove(ambulanceMeshes[ambId]);
            delete ambulanceMeshes[ambId];
        }
    });

    medCenterMeshes.forEach((mesh) => scene.remove(mesh));
    medCenterMeshes = [];
    (data.med_centers || []).forEach((center) => {
        const med = createMedCenter();
        med.position.set(center[0], 0.3, center[1]);
        scene.add(med);
        medCenterMeshes.push(med);
    });

    document.getElementById("victims-saved").textContent = `${data.rescued_victims}/${data.total_victims}`;
    document.getElementById("avg-time").textContent = data.avg_rescue_time > 0 ? `${data.avg_rescue_time.toFixed(2)}s` : "-";
    document.getElementById("risk-exposure").textContent = data.risk_exposure > 0 ? data.risk_exposure.toFixed(3) : "-";
    document.getElementById("csp-assignment").textContent = formatAssignmentPlan(data.csp_assignment);
    document.getElementById("victim-survival").textContent = formatSurvivalList((data.victims || []).filter((victim) => !victim.rescued));
    document.getElementById("alpha-value").textContent = parseFloat(document.getElementById("alpha-slider").value).toFixed(1);
    updateTradeoffBanner();
});

socket.on("route_planned", (data) => {
    const ambId = data.ambulance_id || "A1";
    const ambIndex = (gameState.ambulances || []).findIndex((a) => a.id === ambId);
    const color = routeColors[(ambIndex >= 0 ? ambIndex : 0) % routeColors.length];

    if (routeLines[ambId]) {
        const oldLine = routeLines[ambId];
        oldLine.material.transparent = true;
        oldLine.material.opacity = 0.6;
        fadingRouteLines.push(oldLine);
        delete routeLines[ambId];
    }

    const points = (data.path || []).map((pos) => new THREE.Vector3(pos[0], 0.1, pos[1]));
    if (points.length > 1) {
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        geometry.setDrawRange(0, Math.min(2, points.length));
        const material = new THREE.LineBasicMaterial({ color, linewidth: 2, transparent: true, opacity: 0.95 });
        const line = new THREE.Line(geometry, material);
        scene.add(line);
        routeLines[ambId] = line;
        routeDrawAnimations[ambId] = { line, drawRange: 2, pointsCount: points.length };
        pendingRouteAnimations[ambId] = { pointsCount: points.length };
    }

    appendLogEntry(
        data.is_replan ? "REPLAN" : "ROUTE SELECTION",
        `${ambId} -> ${data.victim_id || "target"} | ${data.algorithm ? data.algorithm.toUpperCase() : "A*"} (alpha=${Number(data.alpha ?? 1).toFixed(1)}), cost=${Number(data.cost || 0).toFixed(2)}, risk=${Number(data.risk_score || 0).toFixed(2)}, opt=${Number(data.optimality_ratio ?? 1).toFixed(3)}, fuzzy=${Number(data.fuzzy_risk_along_path ?? 0).toFixed(3)}. ${data.justification || ""}`,
        data.is_replan ? "REPLAN" : "ROUTE_SELECTION"
    );
});

socket.on("full_rescue_planned", (data) => {
    const planText = Object.entries(data.assignment_plan || {})
        .map(([resource, victims]) => `${resource}: ${Array.isArray(victims) ? victims.join(" -> ") : victims}`)
        .join(" | ");
    const routeText = Object.entries(data.route_summary || {})
        .map(([amb, info]) => `${amb} -> ${info.victim_id} (cost=${Number(info.route_cost || 0).toFixed(2)}, priority=${Number(info.priority || 0).toFixed(3)})`)
        .join(" | ");
    appendLogEntry("ASSIGNMENT", `${planText}. ${routeText}`, "ASSIGNMENT");
});

socket.on("ml_report", (data) => {
    const survival = data.ml_report?.survival || {};
    const risk = data.ml_report?.risk || {};
    appendLogEntry("ML REPORT", `Startup model comparison loaded. Survival models: ${Object.keys(survival).join(", ")}; Risk models: ${Object.keys(risk).join(", ")}.`, "ASSIGNMENT");
});

socket.on("event_triggered", (data) => {
    if (data.type === "block") {
        appendLogEntry("ROAD BLOCKED", `Road blocked at (${data.coords[0]}, ${data.coords[1]}), replanning affected routes.`, "REPLAN");
    }
});

socket.on("victim_added", (data) => {
    appendLogEntry("ASSIGNMENT", `New victim ${data.victim} detected, CSP reallocation triggered.`, "ASSIGNMENT");
});

socket.on("rescue_complete", (data) => {
    appendLogEntry(
        "RESCUE COMPLETE ✅",
        `${data.ambulance_id} rescued ${data.victim_id}. Survival=${Number(data.updated_survival_prob).toFixed(3)}, rescueTime=${Number(data.rescue_time).toFixed(1)}s.`,
        "RESCUE_COMPLETE"
    );
});

document.addEventListener("DOMContentLoaded", init);
