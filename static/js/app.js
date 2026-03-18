/**
 * Face Mask Detection — Live Webcam Detection
 * Captures webcam frames, sends to backend for detection, displays annotated results.
 */

document.addEventListener("DOMContentLoaded", () => {
    // ─── DOM Elements ───
    const startSection = document.getElementById("start-section");
    const liveSection = document.getElementById("live-section");
    const startBtn = document.getElementById("start-btn");
    const pauseBtn = document.getElementById("pause-btn");
    const stopBtn = document.getElementById("stop-btn");
    const pauseText = document.getElementById("pause-text");
    const pauseIcon = document.getElementById("pause-icon");
    const webcamVideo = document.getElementById("webcam-video");
    const captureCanvas = document.getElementById("capture-canvas");
    const resultImage = document.getElementById("result-image");
    const videoOverlay = document.getElementById("video-overlay");
    const videoLoading = document.getElementById("video-loading");
    const statTotal = document.getElementById("stat-total");
    const statMask = document.getElementById("stat-mask");
    const statNoMask = document.getElementById("stat-no-mask");
    const statFps = document.getElementById("stat-fps");
    const detectionsList = document.getElementById("detections-list");

    let stream = null;
    let isRunning = false;
    let isPaused = false;
    let frameCount = 0;
    let lastFpsTime = performance.now();
    let currentFps = 0;
    let isProcessing = false;

    // ─── Start Webcam ───
    startBtn.addEventListener("click", startDetection);

    async function startDetection() {
        try {
            startBtn.textContent = "Starting camera...";
            startBtn.disabled = true;

            // Request webcam access
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: "user",
                },
                audio: false,
            });

            webcamVideo.srcObject = stream;
            await webcamVideo.play();

            // Set canvas size to match video
            captureCanvas.width = webcamVideo.videoWidth || 640;
            captureCanvas.height = webcamVideo.videoHeight || 480;

            // Switch UI
            startSection.style.display = "none";
            liveSection.style.display = "block";

            isRunning = true;
            isPaused = false;

            // Wait a moment for camera to stabilize, then start detection loop
            setTimeout(() => {
                videoLoading.style.display = "none";
                detectLoop();
            }, 1000);

        } catch (err) {
            console.error("Camera access error:", err);
            startBtn.textContent = "Start Webcam Detection";
            startBtn.disabled = false;

            if (err.name === "NotAllowedError") {
                alert("Camera access denied. Please grant camera permission and try again.");
            } else if (err.name === "NotFoundError") {
                alert("No camera found. Please connect a webcam and try again.");
            } else {
                alert("Could not access camera: " + err.message);
            }
        }
    }

    // ─── Detection Loop ───
    async function detectLoop() {
        if (!isRunning) return;
        if (isPaused) {
            requestAnimationFrame(() => setTimeout(detectLoop, 100));
            return;
        }
        if (isProcessing) {
            requestAnimationFrame(() => setTimeout(detectLoop, 50));
            return;
        }

        isProcessing = true;

        try {
            // Capture current frame
            const ctx = captureCanvas.getContext("2d");
            ctx.drawImage(webcamVideo, 0, 0, captureCanvas.width, captureCanvas.height);
            const frameData = captureCanvas.toDataURL("image/jpeg", 0.8);

            // Send to server
            const response = await fetch("/detect", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ frame: frameData }),
            });

            if (!response.ok) throw new Error("Detection failed");

            const data = await response.json();

            // Show annotated result
            resultImage.src = "data:image/jpeg;base64," + data.image;

            // Update stats
            statTotal.textContent = data.total_faces;
            statMask.textContent = data.with_mask;
            statNoMask.textContent = data.without_mask;

            // FPS calculation
            frameCount++;
            const now = performance.now();
            if (now - lastFpsTime >= 1000) {
                currentFps = frameCount;
                frameCount = 0;
                lastFpsTime = now;
                statFps.textContent = currentFps;
            }

            // Update detection log
            updateDetectionLog(data.detections);

        } catch (err) {
            console.error("Detection error:", err);
        }

        isProcessing = false;

        // Continue loop
        if (isRunning) {
            requestAnimationFrame(detectLoop);
        }
    }

    // ─── Update Detection Log ───
    function updateDetectionLog(detections) {
        if (detections.length === 0) {
            detectionsList.innerHTML = `
                <div class="detection-item empty-log">
                    <span>No faces detected</span>
                </div>
            `;
            return;
        }

        detectionsList.innerHTML = "";
        detections.forEach((det, i) => {
            const isMask = det.label === "Mask";
            const cls = isMask ? "mask" : "no-mask";
            const icon = isMask ? "✓" : "✕";

            const item = document.createElement("div");
            item.className = "detection-item";
            item.innerHTML = `
                <span class="detection-badge ${cls}">${icon} ${det.label}</span>
                <div class="detection-bar-wrapper">
                    <div class="detection-bar-label">
                        <span>Face #${i + 1}</span>
                        <span>${det.confidence.toFixed(1)}%</span>
                    </div>
                    <div class="detection-bar">
                        <div class="detection-bar-fill ${cls}" style="width: ${det.confidence}%"></div>
                    </div>
                </div>
            `;
            detectionsList.appendChild(item);
        });
    }

    // ─── Pause / Resume ───
    pauseBtn.addEventListener("click", () => {
        isPaused = !isPaused;

        if (isPaused) {
            pauseText.textContent = "Resume";
            pauseIcon.innerHTML = '<polygon points="5 3 19 12 5 21 5 3"/>';
            videoOverlay.style.display = "flex";
        } else {
            pauseText.textContent = "Pause";
            pauseIcon.innerHTML = '<rect x="6" y="4" width="4" height="16"/><rect x="14" y="4" width="4" height="16"/>';
            videoOverlay.style.display = "none";
        }
    });

    // ─── Stop ───
    stopBtn.addEventListener("click", stopDetection);

    function stopDetection() {
        isRunning = false;
        isPaused = false;

        // Stop webcam
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }

        webcamVideo.srcObject = null;

        // Reset UI
        liveSection.style.display = "none";
        startSection.style.display = "block";
        startBtn.textContent = "Start Webcam Detection";
        startBtn.disabled = false;
        videoLoading.style.display = "flex";
        videoOverlay.style.display = "none";

        // Reset stats
        statTotal.textContent = "0";
        statMask.textContent = "0";
        statNoMask.textContent = "0";
        statFps.textContent = "0";
        detectionsList.innerHTML = `
            <div class="detection-item empty-log">
                <span>Waiting for detections...</span>
            </div>
        `;
    }
});
