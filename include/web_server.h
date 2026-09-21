#ifndef RADAR_WEB_SERVER_H
#define RADAR_WEB_SERVER_H

#include <iostream>
#include <string>
#include <sstream>
#include <fstream>
#include <vector>
#include <atomic>
#include <thread>
#include <mutex>
#include <cstring>
#include <iomanip>
#include <cmath>

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>

// Shared configuration parameters for the radar shader and audio processor
struct RadarConfig {
    std::atomic<float> scanningSpeed{0.4f};    // Scanning speed in rev/sec (matches default 1.0 / 2.5 = 0.4)
    std::atomic<float> decayTime{0.995f};      // Decay persistence factor (default 0.995)
    std::atomic<float> freqMinHz{100.0f};      // Min frequency in Hz (default 100 Hz)
    std::atomic<float> freqMaxHz{10000.0f};    // Max frequency in Hz (default 10000 Hz)
    std::atomic<float> freqRangeMin{0.0f};     // Shader radial freq start (0.0 to 1.0)
    std::atomic<float> freqRangeMax{1.0f};     // Shader radial freq end (0.0 to 1.0)
    std::atomic<float> currentFps{0.0f};       // Measured FPS for live display
    std::atomic<int> frameCount{0};            // Total frames rendered

    std::string toJson() const {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(4);
        ss << "{\n";
        ss << "  \"scanning_speed\": " << scanningSpeed.load() << ",\n";
        ss << "  \"decay_time\": " << decayTime.load() << ",\n";
        ss << "  \"freq_min\": " << freqMinHz.load() << ",\n";
        ss << "  \"freq_max\": " << freqMaxHz.load() << ",\n";
        ss << "  \"freq_range_low\": " << freqRangeMin.load() << ",\n";
        ss << "  \"freq_range_high\": " << freqRangeMax.load() << ",\n";
        ss << "  \"fps\": " << currentFps.load() << ",\n";
        ss << "  \"frame\": " << frameCount.load() << "\n";
        ss << "}";
        return ss.str();
    }
};

// Simple helper to parse float from JSON without external libraries
inline bool extractJsonFloat(const std::string& json, const std::string& key, float& outVal) {
    size_t keyPos = json.find("\"" + key + "\"");
    if (keyPos == std::string::npos) return false;
    size_t colonPos = json.find(':', keyPos);
    if (colonPos == std::string::npos) return false;
    size_t start = json.find_first_of("-0123456789.", colonPos);
    if (start == std::string::npos) return false;
    size_t end = json.find_first_not_of("-0123456789.eE+", start);
    std::string valStr = json.substr(start, (end == std::string::npos ? json.length() : end) - start);
    try {
        outVal = std::stof(valStr);
        return true;
    } catch (...) {
        return false;
    }
}

class RadarWebServer {
public:
    RadarWebServer(RadarConfig& config, int port = 8080)
        : m_config(config), m_port(port), m_running(false), m_serverFd(-1) {}

    ~RadarWebServer() {
        stop();
    }

    bool start() {
        m_serverFd = socket(AF_INET, SOCK_STREAM, 0);
        if (m_serverFd < 0) {
            std::cerr << "[WebServer] Failed to create socket: " << strerror(errno) << std::endl;
            return false;
        }

        int opt = 1;
        if (setsockopt(m_serverFd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt)) < 0) {
            std::cerr << "[WebServer] setsockopt SO_REUSEADDR failed: " << strerror(errno) << std::endl;
        }

        sockaddr_in serverAddr{};
        serverAddr.sin_family = AF_INET;
        serverAddr.sin_addr.s_addr = INADDR_ANY;
        serverAddr.sin_port = htons(m_port);

        if (bind(m_serverFd, (struct sockaddr*)&serverAddr, sizeof(serverAddr)) < 0) {
            std::cerr << "[WebServer] Failed to bind to port " << m_port << ": " << strerror(errno) << std::endl;
            close(m_serverFd);
            m_serverFd = -1;
            return false;
        }

        if (listen(m_serverFd, 16) < 0) {
            std::cerr << "[WebServer] Failed to listen: " << strerror(errno) << std::endl;
            close(m_serverFd);
            m_serverFd = -1;
            return false;
        }

        m_running = true;
        m_serverThread = std::thread(&RadarWebServer::serverLoop, this);

        std::cout << "[WebServer] Radar Web Interface started on port " << m_port << std::endl;
        std::cout << "[WebServer] Open http://localhost:" << m_port << " in your browser" << std::endl;
        return true;
    }

    void stop() {
        if (m_running) {
            m_running = false;
            if (m_serverFd >= 0) {
                shutdown(m_serverFd, SHUT_RDWR);
                close(m_serverFd);
                m_serverFd = -1;
            }
            if (m_serverThread.joinable()) {
                m_serverThread.join();
            }
            std::cout << "[WebServer] Server stopped" << std::endl;
        }
    }

    int getPort() const { return m_port; }

private:
    RadarConfig& m_config;
    int m_port;
    std::atomic<bool> m_running;
    int m_serverFd;
    std::thread m_serverThread;

    std::string getHtmlContent() {
        // First attempt to read from web/index.html
        std::ifstream file("web/index.html");
        if (file.is_open()) {
            std::stringstream ss;
            ss << file.rdbuf();
            return ss.str();
        }
        return getEmbeddedHtml();
    }

    std::string getEmbeddedHtml();

    void serverLoop() {
        while (m_running) {
            struct pollfd pfd;
            pfd.fd = m_serverFd;
            pfd.events = POLLIN;
            pfd.revents = 0;

            int pollRet = poll(&pfd, 1, 500); // 500ms timeout to check m_running
            if (pollRet < 0) {
                if (errno == EINTR) continue;
                break;
            }
            if (pollRet == 0) continue;

            if (pfd.revents & POLLIN) {
                sockaddr_in clientAddr{};
                socklen_t clientLen = sizeof(clientAddr);
                int clientFd = accept(m_serverFd, (struct sockaddr*)&clientAddr, &clientLen);
                if (clientFd >= 0) {
                    // Set client timeout
                    struct timeval tv;
                    tv.tv_sec = 2;
                    tv.tv_usec = 0;
                    setsockopt(clientFd, SOL_SOCKET, SO_RCVTIMEO, (const char*)&tv, sizeof tv);
                    setsockopt(clientFd, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof tv);

                    std::thread([this, clientFd]() {
                        handleClient(clientFd);
                        close(clientFd);
                    }).detach();
                }
            }
        }
    }

    void handleClient(int clientFd) {
        char buffer[8192];
        ssize_t bytesRead = recv(clientFd, buffer, sizeof(buffer) - 1, 0);
        if (bytesRead <= 0) return;
        buffer[bytesRead] = '\0';
        std::string request(buffer, bytesRead);

        // Parse first line: METHOD PATH HTTP/1.1
        size_t firstLineEnd = request.find("\r\n");
        if (firstLineEnd == std::string::npos) return;
        std::string firstLine = request.substr(0, firstLineEnd);

        std::istringstream iss(firstLine);
        std::string method, path, version;
        iss >> method >> path >> version;

        // Strip query params for path matching
        std::string cleanPath = path;
        size_t qPos = cleanPath.find('?');
        if (qPos != std::string::npos) {
            cleanPath = cleanPath.substr(0, qPos);
        }

        // Standard CORS headers
        std::string cors = 
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
            "Access-Control-Allow-Headers: Content-Type\r\n";

        if (method == "OPTIONS") {
            std::string response = "HTTP/1.1 204 No Content\r\n" + cors + "Content-Length: 0\r\n\r\n";
            send(clientFd, response.c_str(), response.length(), 0);
            return;
        }

        if ((method == "GET" || method == "HEAD") && (cleanPath == "/" || cleanPath == "/index.html")) {
            std::string body = getHtmlContent();
            std::ostringstream response;
            response << "HTTP/1.1 200 OK\r\n"
                     << "Content-Type: text/html; charset=utf-8\r\n"
                     << "Content-Length: " << body.length() << "\r\n"
                     << cors
                     << "Connection: close\r\n\r\n";
            if (method == "GET") {
                response << body;
            }
            std::string resStr = response.str();
            send(clientFd, resStr.c_str(), resStr.length(), 0);
            return;
        }

        if ((method == "GET" || method == "HEAD") && (cleanPath == "/api/controls" || cleanPath == "/api/status")) {
            std::string body = m_config.toJson();
            std::ostringstream response;
            response << "HTTP/1.1 200 OK\r\n"
                     << "Content-Type: application/json\r\n"
                     << "Content-Length: " << body.length() << "\r\n"
                     << cors
                     << "Connection: close\r\n\r\n";
            if (method == "GET") {
                response << body;
            }
            std::string resStr = response.str();
            send(clientFd, resStr.c_str(), resStr.length(), 0);
            return;
        }

        if (method == "POST" && cleanPath == "/api/controls") {
            // Find body after \r\n\r\n
            size_t bodyPos = request.find("\r\n\r\n");
            std::string body = (bodyPos != std::string::npos) ? request.substr(bodyPos + 4) : "";

            float val;
            if (extractJsonFloat(body, "scanning_speed", val)) {
                if (val >= 0.0f && val <= 10.0f) {
                    m_config.scanningSpeed.store(val);
                }
            }
            if (extractJsonFloat(body, "decay_time", val)) {
                if (val >= 0.1f && val <= 0.9999f) {
                    m_config.decayTime.store(val);
                }
            }
            if (extractJsonFloat(body, "freq_min", val)) {
                if (val >= 10.0f && val <= 22000.0f) {
                    m_config.freqMinHz.store(val);
                }
            }
            if (extractJsonFloat(body, "freq_max", val)) {
                if (val >= 10.0f && val <= 22000.0f) {
                    m_config.freqMaxHz.store(val);
                }
            }
            if (extractJsonFloat(body, "freq_range_low", val)) {
                if (val >= 0.0f && val <= 1.0f) {
                    m_config.freqRangeMin.store(val);
                }
            }
            if (extractJsonFloat(body, "freq_range_high", val)) {
                if (val >= 0.0f && val <= 1.0f) {
                    m_config.freqRangeMax.store(val);
                }
            }

            std::string resBody = m_config.toJson();
            std::ostringstream response;
            response << "HTTP/1.1 200 OK\r\n"
                     << "Content-Type: application/json\r\n"
                     << "Content-Length: " << resBody.length() << "\r\n"
                     << cors
                     << "Connection: close\r\n\r\n"
                     << resBody;
            std::string resStr = response.str();
            send(clientFd, resStr.c_str(), resStr.length(), 0);
            return;
        }

        // 404 for anything else
        std::string notFound = "HTTP/1.1 404 Not Found\r\n" + cors + "Content-Length: 9\r\n\r\nNot Found";
        send(clientFd, notFound.c_str(), notFound.length(), 0);
    }
};

// Fallback embedded HTML
inline std::string RadarWebServer::getEmbeddedHtml() {
    return R"rawliteral(<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Radar Shader Controller</title>
  <style>
    :root {
      --bg: #0b0f14;
      --card-bg: #121820;
      --accent-green: #00ff88;
      --accent-cyan: #00e5ff;
      --accent-dim: #1e3a34;
      --text: #e0f2ee;
      --text-muted: #6b8f85;
      --border: #1a2c26;
      --slider-track: #172421;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, monospace, sans-serif; }
    body {
      background-color: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 24px 16px;
    }
    .container {
      width: 100%;
      max-width: 900px;
      display: flex;
      flex-direction: column;
      gap: 20px;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 16px 24px;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      box-shadow: 0 4px 20px rgba(0, 255, 136, 0.05);
    }
    .title-area h1 {
      font-size: 1.3rem;
      font-weight: 700;
      letter-spacing: 2px;
      color: var(--accent-green);
      text-transform: uppercase;
      display: flex;
      align-items: center;
      gap: 10px;
    }
    .badge {
      font-size: 0.75rem;
      padding: 4px 10px;
      border-radius: 20px;
      font-weight: bold;
      letter-spacing: 1px;
    }
    .badge-online {
      background: rgba(0, 255, 136, 0.15);
      color: var(--accent-green);
      border: 1px solid var(--accent-green);
    }
    .badge-offline {
      background: rgba(255, 68, 68, 0.15);
      color: #ff4444;
      border: 1px solid #ff4444;
    }
    .grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }
    @media (max-width: 768px) {
      .grid { grid-template-columns: 1fr; }
    }
    .card {
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .card-title {
      font-size: 0.9rem;
      font-weight: 600;
      color: var(--accent-cyan);
      letter-spacing: 1.5px;
      text-transform: uppercase;
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid var(--border);
      padding-bottom: 8px;
    }
    .radar-preview-card {
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 280px;
    }
    canvas#radarCanvas {
      width: 240px;
      height: 240px;
      border-radius: 50%;
      background: #081010;
      border: 2px solid var(--accent-dim);
      box-shadow: 0 0 25px rgba(0, 255, 136, 0.1);
    }
    .control-group {
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .control-label {
      display: flex;
      justify-content: space-between;
      font-size: 0.85rem;
      color: var(--text-muted);
    }
    .control-value {
      color: var(--accent-green);
      font-family: monospace;
      font-weight: 700;
      font-size: 0.95rem;
    }
    input[type=range] {
      -webkit-appearance: none;
      width: 100%;
      height: 6px;
      border-radius: 3px;
      background: var(--slider-track);
      outline: none;
    }
    input[type=range]::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 18px;
      height: 18px;
      border-radius: 50%;
      background: var(--accent-green);
      cursor: pointer;
      box-shadow: 0 0 8px var(--accent-green);
      transition: transform 0.1s;
    }
    input[type=range]::-webkit-slider-thumb:hover {
      transform: scale(1.2);
    }
    .btn-group {
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 4px;
    }
    button.btn {
      background: var(--card-bg);
      border: 1px solid var(--border);
      color: var(--text);
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 0.75rem;
      cursor: pointer;
      transition: all 0.2s;
    }
    button.btn:hover {
      border-color: var(--accent-green);
      color: var(--accent-green);
      background: rgba(0, 255, 136, 0.05);
    }
    button.btn-primary {
      background: rgba(0, 255, 136, 0.1);
      border-color: var(--accent-green);
      color: var(--accent-green);
    }
    .stats-bar {
      display: flex;
      justify-content: space-around;
      padding: 12px;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      font-family: monospace;
      font-size: 0.85rem;
      color: var(--text-muted);
    }
    .stat-val {
      color: var(--accent-cyan);
      font-weight: bold;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="title-area">
        <h1>RADAR SHADER CONTROLS</h1>
      </div>
      <div id="statusBadge" class="badge badge-offline">CONNECTING...</div>
    </header>

    <div class="stats-bar">
      <div>FPS: <span id="statFps" class="stat-val">--</span></div>
      <div>FRAME: <span id="statFrame" class="stat-val">--</span></div>
      <div>SWEEP: <span id="statSweep" class="stat-val">0.40 rev/s</span></div>
      <div>DECAY: <span id="statDecay" class="stat-val">0.9950</span></div>
      <div>RANGE: <span id="statFreq" class="stat-val">100 - 10000 Hz</span></div>
    </div>

    <div class="grid">
      <!-- Radar Simulation Canvas -->
      <div class="card radar-preview-card">
        <canvas id="radarCanvas" width="240" height="240"></canvas>
        <div style="font-size:0.8rem; color:var(--text-muted); margin-top:8px;">Live Parameter Visualizer</div>
      </div>

      <!-- Radar Motion & Decay -->
      <div class="card">
        <div class="card-title">
          <span>Beam & Persistence</span>
        </div>

        <!-- Scanning Speed -->
        <div class="control-group">
          <div class="control-label">
            <span>Scanning Speed</span>
            <span id="speedVal" class="control-value">0.40 rev/s</span>
          </div>
          <input type="range" id="speedSlider" min="0.05" max="3.00" step="0.05" value="0.40">
          <div class="btn-group">
            <button class="btn" onclick="setSpeed(0.1)">0.1x Slow</button>
            <button class="btn" onclick="setSpeed(0.4)">0.4x Default</button>
            <button class="btn" onclick="setSpeed(1.0)">1.0x 1 rev/s</button>
            <button class="btn" onclick="setSpeed(2.0)">2.0x Fast</button>
          </div>
        </div>

        <!-- Decay Time -->
        <div class="control-group">
          <div class="control-label">
            <span>Decay Persistence Factor</span>
            <span id="decayVal" class="control-value">0.9950 (~2.3s)</span>
          </div>
          <input type="range" id="decaySlider" min="0.8000" max="0.9995" step="0.0005" value="0.9950">
          <div class="btn-group">
            <button class="btn" onclick="setDecay(0.9000)">Fast (0.90)</button>
            <button class="btn" onclick="setDecay(0.9700)">Med (0.97)</button>
            <button class="btn" onclick="setDecay(0.9950)">Default (0.995)</button>
            <button class="btn" onclick="setDecay(0.9990)">Ghost (0.999)</button>
          </div>
        </div>
      </div>
    </div>

    <!-- Frequency Range Card -->
    <div class="card">
      <div class="card-title">
        <span>Audio FFT & Shader Frequency Range</span>
      </div>

      <div class="grid" style="gap:16px;">
        <div class="control-group">
          <div class="control-label">
            <span>Min Audible Frequency</span>
            <span id="freqMinVal" class="control-value">100 Hz</span>
          </div>
          <input type="range" id="freqMinSlider" min="20" max="2000" step="10" value="100">
        </div>

        <div class="control-group">
          <div class="control-label">
            <span>Max Audible Frequency</span>
            <span id="freqMaxVal" class="control-value">10000 Hz</span>
          </div>
          <input type="range" id="freqMaxSlider" min="500" max="20000" step="100" value="10000">
        </div>
      </div>

      <div class="control-group" style="margin-top: 4px;">
        <div class="control-label">
          <span>Shader Radial Span (Inner - Outer)</span>
          <span id="spanVal" class="control-value">0% - 100%</span>
        </div>
        <div style="display:flex; gap:10px;">
          <input type="range" id="spanMinSlider" min="0.0" max="0.9" step="0.05" value="0.0" style="flex:1;">
          <input type="range" id="spanMaxSlider" min="0.1" max="1.0" step="0.05" value="1.0" style="flex:1;">
        </div>
      </div>

      <div class="btn-group">
        <span style="font-size:0.75rem; color:var(--text-muted); align-self:center; margin-right:6px;">Presets:</span>
        <button class="btn" onclick="setFreqBand(20, 20000)">Full Spectrum (20Hz - 20kHz)</button>
        <button class="btn" onclick="setFreqBand(20, 300)">Sub & Bass (20Hz - 300Hz)</button>
        <button class="btn" onclick="setFreqBand(250, 4000)">Voice & Mids (250Hz - 4kHz)</button>
        <button class="btn" onclick="setFreqBand(4000, 18000)">Highs & Air (4kHz - 18kHz)</button>
        <button class="btn" onclick="setFreqBand(100, 10000)">Default (100Hz - 10kHz)</button>
      </div>
    </div>

    <!-- Presets Card -->
    <div class="card">
      <div class="card-title">
        <span>Global Shader Presets</span>
      </div>
      <div class="btn-group">
        <button class="btn btn-primary" onclick="applyPreset('default')">Default Radar</button>
        <button class="btn" onclick="applyPreset('sonar')">Deep Sub Sonar</button>
        <button class="btn" onclick="applyPreset('strobe')">Fast Strobe Visualizer</button>
        <button class="btn" onclick="applyPreset('phosphor')">Long Phosphor Glow</button>
        <button class="btn" onclick="applyPreset('matrix')">High Sweep Matrix</button>
      </div>
    </div>
  </div>

  <script>
    let state = {
      scanning_speed: 0.40,
      decay_time: 0.9950,
      freq_min: 100,
      freq_max: 10000,
      freq_range_low: 0.0,
      freq_range_high: 1.0,
      fps: 0,
      frame: 0
    };

    const speedSlider = document.getElementById('speedSlider');
    const decaySlider = document.getElementById('decaySlider');
    const freqMinSlider = document.getElementById('freqMinSlider');
    const freqMaxSlider = document.getElementById('freqMaxSlider');
    const spanMinSlider = document.getElementById('spanMinSlider');
    const spanMaxSlider = document.getElementById('spanMaxSlider');

    const speedVal = document.getElementById('speedVal');
    const decayVal = document.getElementById('decayVal');
    const freqMinVal = document.getElementById('freqMinVal');
    const freqMaxVal = document.getElementById('freqMaxVal');
    const spanVal = document.getElementById('spanVal');

    const statFps = document.getElementById('statFps');
    const statFrame = document.getElementById('statFrame');
    const statSweep = document.getElementById('statSweep');
    const statDecay = document.getElementById('statDecay');
    const statFreq = document.getElementById('statFreq');
    const statusBadge = document.getElementById('statusBadge');

    let isSending = false;
    let pendingUpdate = false;

    function calculateHalfLife(decay) {
      if (decay >= 0.9999) return 'inf';
      if (decay <= 0) return '0s';
      const halfLifeSec = Math.log(0.5) / (60 * Math.log(decay));
      return halfLifeSec < 1 ? (halfLifeSec * 1000).toFixed(0) + 'ms' : halfLifeSec.toFixed(1) + 's';
    }

    function updateUI(s) {
      speedVal.innerText = Number(s.scanning_speed).toFixed(2) + ' rev/s';
      decayVal.innerText = Number(s.decay_time).toFixed(4) + ' (~' + calculateHalfLife(s.decay_time) + ')';
      freqMinVal.innerText = Math.round(s.freq_min) + ' Hz';
      freqMaxVal.innerText = Math.round(s.freq_max) + ' Hz';
      spanVal.innerText = Math.round(s.freq_range_low * 100) + '% - ' + Math.round(s.freq_range_high * 100) + '%';

      statSweep.innerText = Number(s.scanning_speed).toFixed(2) + ' rev/s';
      statDecay.innerText = Number(s.decay_time).toFixed(4);
      statFreq.innerText = Math.round(s.freq_min) + ' - ' + Math.round(s.freq_max) + ' Hz';

      if (s.fps > 0) statFps.innerText = s.fps.toFixed(1);
      if (s.frame > 0) statFrame.innerText = s.frame;
    }

    function syncSlidersFromState() {
      speedSlider.value = state.scanning_speed;
      decaySlider.value = state.decay_time;
      freqMinSlider.value = state.freq_min;
      freqMaxSlider.value = state.freq_max;
      spanMinSlider.value = state.freq_range_low;
      spanMaxSlider.value = state.freq_range_high;
      updateUI(state);
    }

    async function sendControls() {
      if (isSending) {
        pendingUpdate = true;
        return;
      }
      isSending = true;
      try {
        const payload = {
          scanning_speed: parseFloat(state.scanning_speed),
          decay_time: parseFloat(state.decay_time),
          freq_min: parseFloat(state.freq_min),
          freq_max: parseFloat(state.freq_max),
          freq_range_low: parseFloat(state.freq_range_low),
          freq_range_high: parseFloat(state.freq_range_high)
        };
        const res = await fetch('/api/controls', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (res.ok) {
          const data = await res.json();
          statusBadge.className = 'badge badge-online';
          statusBadge.innerText = 'ONLINE';
          if (data.fps) statFps.innerText = Number(data.fps).toFixed(1);
          if (data.frame) statFrame.innerText = data.frame;
        }
      } catch (err) {
        statusBadge.className = 'badge badge-offline';
        statusBadge.innerText = 'DISCONNECTED';
      } finally {
        isSending = false;
        if (pendingUpdate) {
          pendingUpdate = false;
          sendControls();
        }
      }
    }

    function onSliderChange() {
      state.scanning_speed = parseFloat(speedSlider.value);
      state.decay_time = parseFloat(decaySlider.value);
      state.freq_min = parseFloat(freqMinSlider.value);
      state.freq_max = parseFloat(freqMaxSlider.value);
      state.freq_range_low = parseFloat(spanMinSlider.value);
      state.freq_range_high = parseFloat(spanMaxSlider.value);

      if (state.freq_min >= state.freq_max) {
        state.freq_max = state.freq_min + 100;
        freqMaxSlider.value = state.freq_max;
      }
      if (state.freq_range_low >= state.freq_range_high) {
        state.freq_range_high = Math.min(1.0, state.freq_range_low + 0.05);
        spanMaxSlider.value = state.freq_range_high;
      }

      updateUI(state);
      sendControls();
    }

    [speedSlider, decaySlider, freqMinSlider, freqMaxSlider, spanMinSlider, spanMaxSlider].forEach(el => {
      el.addEventListener('input', onSliderChange);
    });

    function setSpeed(v) {
      speedSlider.value = v;
      onSliderChange();
    }
    function setDecay(v) {
      decaySlider.value = v;
      onSliderChange();
    }
    function setFreqBand(min, max) {
      freqMinSlider.value = min;
      freqMaxSlider.value = max;
      onSliderChange();
    }

    function applyPreset(name) {
      if (name === 'default') {
        state.scanning_speed = 0.40;
        state.decay_time = 0.9950;
        state.freq_min = 100;
        state.freq_max = 10000;
        state.freq_range_low = 0.0;
        state.freq_range_high = 1.0;
      } else if (name === 'sonar') {
        state.scanning_speed = 0.15;
        state.decay_time = 0.9980;
        state.freq_min = 20;
        state.freq_max = 500;
        state.freq_range_low = 0.0;
        state.freq_range_high = 1.0;
      } else if (name === 'strobe') {
        state.scanning_speed = 1.80;
        state.decay_time = 0.9100;
        state.freq_min = 60;
        state.freq_max = 14000;
        state.freq_range_low = 0.0;
        state.freq_range_high = 1.0;
      } else if (name === 'phosphor') {
        state.scanning_speed = 0.25;
        state.decay_time = 0.9990;
        state.freq_min = 50;
        state.freq_max = 12000;
        state.freq_range_low = 0.0;
        state.freq_range_high = 1.0;
      } else if (name === 'matrix') {
        state.scanning_speed = 0.90;
        state.decay_time = 0.9800;
        state.freq_min = 200;
        state.freq_max = 8000;
        state.freq_range_low = 0.1;
        state.freq_range_high = 0.9;
      }
      syncSlidersFromState();
      sendControls();
    }

    async function pollStatus() {
      try {
        const res = await fetch('/api/status');
        if (res.ok) {
          const data = await res.json();
          statusBadge.className = 'badge badge-online';
          statusBadge.innerText = 'ONLINE';
          if (!isSending) {
            state.fps = data.fps;
            state.frame = data.frame;
            updateUI(state);
          }
        }
      } catch (err) {
        statusBadge.className = 'badge badge-offline';
        statusBadge.innerText = 'DISCONNECTED';
      }
    }

    // Live canvas radar visualization matching parameters
    const canvas = document.getElementById('radarCanvas');
    const ctx = canvas.getContext('2d');
    let currentAngle = 0;
    let lastTime = performance.now();

    function drawRadar() {
      const now = performance.now();
      const dt = (now - lastTime) / 1000;
      lastTime = now;

      // Update radar angle based on scanning_speed
      currentAngle += (state.scanning_speed * 2 * Math.PI) * dt;

      const w = canvas.width;
      const h = canvas.height;
      const cx = w / 2;
      const cy = h / 2;
      const radius = w / 2 - 10;

      // Fade canvas using decay_time
      ctx.fillStyle = 'rgba(8, 16, 16, 0.08)';
      ctx.fillRect(0, 0, w, h);

      // Concentric circles
      ctx.strokeStyle = 'rgba(0, 255, 136, 0.15)';
      ctx.lineWidth = 1;
      for (let r = radius / 4; r <= radius; r += radius / 4) {
        ctx.beginPath();
        ctx.arc(cx, cy, r, 0, 2 * Math.PI);
        ctx.stroke();
      }

      // Crosshairs
      ctx.beginPath();
      ctx.moveTo(cx, cy - radius);
      ctx.lineTo(cx, cy + radius);
      ctx.moveTo(cx - radius, cy);
      ctx.lineTo(cx + radius, cy);
      ctx.stroke();

      // Radar sweep beam
      const beamX = cx + Math.cos(currentAngle) * radius;
      const beamY = cy + Math.sin(currentAngle) * radius;

      const grad = ctx.createRadialGradient(cx, cy, 5, cx, cy, radius);
      grad.addColorStop(0, 'rgba(0, 255, 136, 0.9)');
      grad.addColorStop(1, 'rgba(0, 229, 255, 0.3)');

      ctx.strokeStyle = grad;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(beamX, beamY);
      ctx.stroke();

      requestAnimationFrame(drawRadar);
    }

    // Initialize
    syncSlidersFromState();
    pollStatus();
    setInterval(pollStatus, 1500);
    requestAnimationFrame(drawRadar);
  </script>
</body>
</html>
)rawliteral";
}

#endif // RADAR_WEB_SERVER_H
