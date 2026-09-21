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

    // Frequency scaling: 0 = log, 1 = mel, 2 = linear
    std::atomic<int> scaleType{0};

    // Autotuning for scanning speed (musical rhythm sync)
    std::atomic<bool> autoSpeed{false};
    std::atomic<int> rhythmDivision{2};       // 0: 1/4 note (1 beat), 1: 1/2 note, 2: 1 bar (default), 3: 2 bars, 4: 4 bars
    std::atomic<float> detectedBpm{120.0f};

    // Autotuning for frequency range / scaling
    std::atomic<bool> autoFreq{false};
    std::atomic<float> autoGain{1.0f};

    std::string scaleName() const {
        int s = scaleType.load();
        if (s == 1) return "mel";
        if (s == 2) return "linear";
        return "log";
    }

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
        ss << "  \"scale_type\": " << scaleType.load() << ",\n";
        ss << "  \"scale\": \"" << scaleName() << "\",\n";
        ss << "  \"auto_speed\": " << (autoSpeed.load() ? "true" : "false") << ",\n";
        ss << "  \"rhythm_division\": " << rhythmDivision.load() << ",\n";
        ss << "  \"detected_bpm\": " << detectedBpm.load() << ",\n";
        ss << "  \"auto_freq\": " << (autoFreq.load() ? "true" : "false") << ",\n";
        ss << "  \"auto_gain\": " << autoGain.load() << ",\n";
        ss << "  \"fps\": " << currentFps.load() << ",\n";
        ss << "  \"frame\": " << frameCount.load() << "\n";
        ss << "}";
        return ss.str();
    }
};

// JSON parsing helpers
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

inline bool extractJsonInt(const std::string& json, const std::string& key, int& outVal) {
    size_t keyPos = json.find("\"" + key + "\"");
    if (keyPos == std::string::npos) return false;
    size_t colonPos = json.find(':', keyPos);
    if (colonPos == std::string::npos) return false;
    size_t start = json.find_first_of("-0123456789", colonPos);
    if (start == std::string::npos) return false;
    size_t end = json.find_first_not_of("-0123456789", start);
    std::string valStr = json.substr(start, (end == std::string::npos ? json.length() : end) - start);
    try {
        outVal = std::stoi(valStr);
        return true;
    } catch (...) {
        return false;
    }
}

inline bool extractJsonBool(const std::string& json, const std::string& key, bool& outVal) {
    size_t keyPos = json.find("\"" + key + "\"");
    if (keyPos == std::string::npos) return false;
    size_t colonPos = json.find(':', keyPos);
    if (colonPos == std::string::npos) return false;
    size_t valPos = json.find_first_not_of(" \t\r\n", colonPos + 1);
    if (valPos == std::string::npos) return false;
    if (json.substr(valPos, 4) == "true" || json[valPos] == '1') {
        outVal = true;
        return true;
    }
    if (json.substr(valPos, 5) == "false" || json[valPos] == '0') {
        outVal = false;
        return true;
    }
    return false;
}

inline bool extractJsonString(const std::string& json, const std::string& key, std::string& outVal) {
    size_t keyPos = json.find("\"" + key + "\"");
    if (keyPos == std::string::npos) return false;
    size_t colonPos = json.find(':', keyPos);
    if (colonPos == std::string::npos) return false;
    size_t quoteStart = json.find('\"', colonPos);
    if (quoteStart == std::string::npos) return false;
    size_t quoteEnd = json.find('\"', quoteStart + 1);
    if (quoteEnd == std::string::npos) return false;
    outVal = json.substr(quoteStart + 1, quoteEnd - quoteStart - 1);
    return true;
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
        // Read from web/index.html if available
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

            int pollRet = poll(&pfd, 1, 500);
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

        size_t firstLineEnd = request.find("\r\n");
        if (firstLineEnd == std::string::npos) return;
        std::string firstLine = request.substr(0, firstLineEnd);

        std::istringstream iss(firstLine);
        std::string method, path, version;
        iss >> method >> path >> version;

        std::string cleanPath = path;
        size_t qPos = cleanPath.find('?');
        if (qPos != std::string::npos) {
            cleanPath = cleanPath.substr(0, qPos);
        }

        std::string cors = 
            "Access-Control-Allow-Origin: *\r\n"
            "Access-Control-Allow-Methods: GET, POST, OPTIONS, HEAD\r\n"
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
            size_t bodyPos = request.find("\r\n\r\n");
            std::string body = (bodyPos != std::string::npos) ? request.substr(bodyPos + 4) : "";

            float val;
            int intVal;
            bool boolVal;
            std::string strVal;

            if (extractJsonFloat(body, "scanning_speed", val)) {
                if (val >= 0.01f && val <= 10.0f) {
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

            // Scale setting
            if (extractJsonString(body, "scale", strVal)) {
                if (strVal == "mel") m_config.scaleType.store(1);
                else if (strVal == "linear") m_config.scaleType.store(2);
                else if (strVal == "log") m_config.scaleType.store(0);
            } else if (extractJsonInt(body, "scale_type", intVal)) {
                if (intVal >= 0 && intVal <= 2) {
                    m_config.scaleType.store(intVal);
                }
            }

            // Auto speed setting
            if (extractJsonBool(body, "auto_speed", boolVal)) {
                m_config.autoSpeed.store(boolVal);
            }
            if (extractJsonInt(body, "rhythm_division", intVal)) {
                if (intVal >= 0 && intVal <= 4) {
                    m_config.rhythmDivision.store(intVal);
                }
            }

            // Auto freq setting
            if (extractJsonBool(body, "auto_freq", boolVal)) {
                m_config.autoFreq.store(boolVal);
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
  <title>Radar Shader Controls</title>
  <style>
    :root {
      --bg: #0a0e13;
      --card-bg: #121820;
      --card-border: #1a262f;
      --accent: #00ff88;
      --accent-rgb: 0, 255, 136;
      --text: #e2edf2;
      --text-muted: #6f8898;
      --track: #172028;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace, sans-serif; }
    body {
      background-color: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 24px 16px;
    }
    .panel {
      width: 100%;
      max-width: 520px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 6px;
      border-bottom: 1px solid var(--card-border);
    }
    .title {
      font-size: 0.95rem;
      font-weight: 700;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: var(--accent);
    }
    .conn-dot {
      width: 8px;
      height: 8px;
      border-radius: 50%;
      background: #444;
      display: inline-block;
      transition: background 0.3s, box-shadow 0.3s;
    }
    .conn-dot.online {
      background: var(--accent);
      box-shadow: 0 0 8px var(--accent);
    }
    .section {
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 16px 18px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }
    .section-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .section-title {
      font-size: 0.8rem;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 1.2px;
      color: var(--text-muted);
    }
    .val {
      font-family: monospace;
      font-size: 0.85rem;
      font-weight: 700;
      color: var(--accent);
    }
    input[type=range] {
      -webkit-appearance: none;
      width: 100%;
      height: 6px;
      border-radius: 3px;
      background: var(--track);
      outline: none;
    }
    input[type=range]::-webkit-slider-thumb {
      -webkit-appearance: none;
      width: 16px;
      height: 16px;
      border-radius: 50%;
      background: var(--accent);
      cursor: pointer;
      box-shadow: 0 0 6px var(--accent);
    }
    .btn-row {
      display: flex;
      gap: 6px;
      margin-top: 2px;
    }
    button.btn {
      background: #0f151c;
      border: 1px solid var(--card-border);
      color: var(--text);
      padding: 6px 10px;
      border-radius: 5px;
      font-size: 0.74rem;
      cursor: pointer;
      transition: all 0.15s;
      flex: 1;
      text-align: center;
    }
    button.btn:hover {
      border-color: var(--accent);
      color: var(--accent);
    }
    button.btn.active {
      background: rgba(var(--accent-rgb), 0.15);
      border-color: var(--accent);
      color: var(--accent);
      font-weight: 600;
    }
    button.toggle-btn {
      background: #0f151c;
      border: 1px solid var(--card-border);
      color: var(--text-muted);
      padding: 6px 12px;
      border-radius: 5px;
      font-size: 0.74rem;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      transition: all 0.2s;
    }
    button.toggle-btn.active {
      background: rgba(var(--accent-rgb), 0.15);
      border-color: var(--accent);
      color: var(--accent);
      box-shadow: 0 0 10px rgba(var(--accent-rgb), 0.2);
    }
    .meta-row {
      display: flex;
      justify-content: space-between;
      font-size: 0.72rem;
      color: var(--text-muted);
      font-family: monospace;
      margin-top: -2px;
    }
  </style>
</head>
<body>
  <div class="panel">
    <header>
      <div class="title">Radar Shader Controls</div>
      <div id="connDot" class="conn-dot" title="Connection status"></div>
    </header>

    <!-- Scanning Speed -->
    <div class="section">
      <div class="section-head">
        <span class="section-title">Scanning Speed</span>
        <span id="speedVal" class="val">0.40 rev/s</span>
      </div>
      <input type="range" id="speedSlider" min="0.02" max="3.00" step="0.01" value="0.40">
      
      <div style="display:flex; justify-content:space-between; align-items:center; gap:8px;">
        <button id="autoSpeedBtn" class="toggle-btn" onclick="toggleAutoSpeed()">
          ⚡ Auto Rhythm Sync
        </button>
        <div id="rhythmInfo" class="meta-row" style="margin:0;">
          <span id="bpmLabel">BPM: --</span>
        </div>
      </div>

      <div class="btn-row">
        <button class="btn" id="divBtn0" onclick="setRhythmDiv(0)">1/4 (Beat)</button>
        <button class="btn" id="divBtn1" onclick="setRhythmDiv(1)">1/2 (2 Beats)</button>
        <button class="btn active" id="divBtn2" onclick="setRhythmDiv(2)">1 Bar</button>
        <button class="btn" id="divBtn3" onclick="setRhythmDiv(3)">2 Bars</button>
        <button class="btn" id="divBtn4" onclick="setRhythmDiv(4)">4 Bars</button>
      </div>
    </div>

    <!-- Decay Time -->
    <div class="section">
      <div class="section-head">
        <span class="section-title">Decay Persistence</span>
        <span id="decayVal" class="val">0.9950</span>
      </div>
      <input type="range" id="decaySlider" min="0.8000" max="0.9995" step="0.0005" value="0.9950">
      <div class="btn-row">
        <button class="btn" onclick="setDecay(0.9200)">Fast</button>
        <button class="btn" onclick="setDecay(0.9700)">Medium</button>
        <button class="btn active" id="decayDefBtn" onclick="setDecay(0.9950)">Default</button>
        <button class="btn" onclick="setDecay(0.9990)">Long Glow</button>
      </div>
    </div>

    <!-- Frequency Scale Mode (Mel / Log / Linear) -->
    <div class="section">
      <div class="section-head">
        <span class="section-title">Frequency Scale</span>
        <span id="scaleLabel" class="val">Logarithmic</span>
      </div>
      <div class="btn-row">
        <button class="btn active" id="scaleBtnLog" onclick="setScale('log')">Log</button>
        <button class="btn" id="scaleBtnMel" onclick="setScale('mel')">Mel</button>
        <button class="btn" id="scaleBtnLin" onclick="setScale('linear')">Linear</button>
      </div>
    </div>

    <!-- Frequency Range & Scaling Autotune -->
    <div class="section">
      <div class="section-head">
        <span class="section-title">Frequency Range</span>
        <span id="freqRangeVal" class="val">100 - 10000 Hz</span>
      </div>

      <div style="display:flex; flex-direction:column; gap:4px;">
        <div class="meta-row"><span>Min Freq</span><span id="minFreqVal">100 Hz</span></div>
        <input type="range" id="freqMinSlider" min="20" max="2000" step="10" value="100">
      </div>

      <div style="display:flex; flex-direction:column; gap:4px; margin-top:4px;">
        <div class="meta-row"><span>Max Freq</span><span id="maxFreqVal">10000 Hz</span></div>
        <input type="range" id="freqMaxSlider" min="200" max="20000" step="100" value="10000">
      </div>

      <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; margin-top:4px;">
        <button id="autoFreqBtn" class="toggle-btn" onclick="toggleAutoFreq()">
          ⚡ Auto Freq & Scaling
        </button>
        <span id="gainLabel" class="meta-row" style="margin:0;">Gain: 1.0x</span>
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
      scale: "log",
      scale_type: 0,
      auto_speed: false,
      rhythm_division: 2,
      detected_bpm: 120.0,
      auto_freq: false,
      auto_gain: 1.0
    };

    const connDot = document.getElementById('connDot');
    const speedSlider = document.getElementById('speedSlider');
    const decaySlider = document.getElementById('decaySlider');
    const freqMinSlider = document.getElementById('freqMinSlider');
    const freqMaxSlider = document.getElementById('freqMaxSlider');

    const speedVal = document.getElementById('speedVal');
    const decayVal = document.getElementById('decayVal');
    const freqRangeVal = document.getElementById('freqRangeVal');
    const minFreqVal = document.getElementById('minFreqVal');
    const maxFreqVal = document.getElementById('maxFreqVal');
    const scaleLabel = document.getElementById('scaleLabel');
    const bpmLabel = document.getElementById('bpmLabel');
    const gainLabel = document.getElementById('gainLabel');

    const autoSpeedBtn = document.getElementById('autoSpeedBtn');
    const autoFreqBtn = document.getElementById('autoFreqBtn');

    let isSending = false;
    let pendingUpdate = false;

    function updateUI(s) {
      speedVal.innerText = Number(s.scanning_speed).toFixed(2) + ' rev/s';
      decayVal.innerText = Number(s.decay_time).toFixed(4);
      minFreqVal.innerText = Math.round(s.freq_min) + ' Hz';
      maxFreqVal.innerText = Math.round(s.freq_max) + ' Hz';
      freqRangeVal.innerText = Math.round(s.freq_min) + ' - ' + Math.round(s.freq_max) + ' Hz';

      if (!autoSpeedBtn.matches(':active') && s.auto_speed !== undefined) {
        autoSpeedBtn.classList.toggle('active', !!s.auto_speed);
      }
      if (!autoFreqBtn.matches(':active') && s.auto_freq !== undefined) {
        autoFreqBtn.classList.toggle('active', !!s.auto_freq);
      }

      if (s.detected_bpm > 0) {
        bpmLabel.innerText = 'BPM: ' + Math.round(s.detected_bpm);
      }
      if (s.auto_gain) {
        gainLabel.innerText = 'Gain: ' + Number(s.auto_gain).toFixed(1) + 'x';
      }

      const curScale = s.scale || (s.scale_type === 1 ? 'mel' : s.scale_type === 2 ? 'linear' : 'log');
      scaleLabel.innerText = curScale === 'mel' ? 'Mel' : curScale === 'linear' ? 'Linear' : 'Logarithmic';
      document.getElementById('scaleBtnLog').classList.toggle('active', curScale === 'log');
      document.getElementById('scaleBtnMel').classList.toggle('active', curScale === 'mel');
      document.getElementById('scaleBtnLin').classList.toggle('active', curScale === 'linear');

      for (let i = 0; i <= 4; i++) {
        const btn = document.getElementById('divBtn' + i);
        if (btn) btn.classList.toggle('active', s.rhythm_division === i);
      }
    }

    function syncSliders() {
      speedSlider.value = state.scanning_speed;
      decaySlider.value = state.decay_time;
      freqMinSlider.value = state.freq_min;
      freqMaxSlider.value = state.freq_max;
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
          scale: state.scale,
          scale_type: parseInt(state.scale_type),
          auto_speed: !!state.auto_speed,
          rhythm_division: parseInt(state.rhythm_division),
          auto_freq: !!state.auto_freq
        };
        const res = await fetch('/api/controls', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (res.ok) {
          const data = await res.json();
          connDot.className = 'conn-dot online';
          if (data.detected_bpm) state.detected_bpm = data.detected_bpm;
          if (data.auto_gain) state.auto_gain = data.auto_gain;
          if (data.scanning_speed && state.auto_speed) {
            state.scanning_speed = data.scanning_speed;
            speedSlider.value = data.scanning_speed;
          }
          if (data.freq_min && state.auto_freq) {
            state.freq_min = data.freq_min;
            state.freq_max = data.freq_max;
            freqMinSlider.value = data.freq_min;
            freqMaxSlider.value = data.freq_max;
          }
          updateUI(state);
        }
      } catch (err) {
        connDot.className = 'conn-dot';
      } finally {
        isSending = false;
        if (pendingUpdate) {
          pendingUpdate = false;
          sendControls();
        }
      }
    }

    speedSlider.addEventListener('input', () => {
      state.scanning_speed = parseFloat(speedSlider.value);
      state.auto_speed = false;
      autoSpeedBtn.classList.remove('active');
      updateUI(state);
      sendControls();
    });

    decaySlider.addEventListener('input', () => {
      state.decay_time = parseFloat(decaySlider.value);
      updateUI(state);
      sendControls();
    });

    freqMinSlider.addEventListener('input', () => {
      state.freq_min = parseFloat(freqMinSlider.value);
      if (state.freq_min >= state.freq_max) {
        state.freq_max = state.freq_min + 50;
        freqMaxSlider.value = state.freq_max;
      }
      state.auto_freq = false;
      autoFreqBtn.classList.remove('active');
      updateUI(state);
      sendControls();
    });

    freqMaxSlider.addEventListener('input', () => {
      state.freq_max = parseFloat(freqMaxSlider.value);
      if (state.freq_max <= state.freq_min) {
        state.freq_min = Math.max(20, state.freq_max - 50);
        freqMinSlider.value = state.freq_min;
      }
      state.auto_freq = false;
      autoFreqBtn.classList.remove('active');
      updateUI(state);
      sendControls();
    });

    function setDecay(v) {
      decaySlider.value = v;
      state.decay_time = v;
      updateUI(state);
      sendControls();
    }

    function setScale(name) {
      state.scale = name;
      state.scale_type = name === 'mel' ? 1 : name === 'linear' ? 2 : 0;
      updateUI(state);
      sendControls();
    }

    function toggleAutoSpeed() {
      state.auto_speed = !state.auto_speed;
      autoSpeedBtn.classList.toggle('active', state.auto_speed);
      sendControls();
    }

    function setRhythmDiv(div) {
      state.rhythm_division = div;
      updateUI(state);
      sendControls();
    }

    function toggleAutoFreq() {
      state.auto_freq = !state.auto_freq;
      autoFreqBtn.classList.toggle('active', state.auto_freq);
      sendControls();
    }

    async function pollStatus() {
      try {
        const res = await fetch('/api/status');
        if (res.ok) {
          const data = await res.json();
          connDot.className = 'conn-dot online';
          if (!isSending) {
            if (data.auto_speed) {
              state.scanning_speed = data.scanning_speed;
              speedSlider.value = data.scanning_speed;
            }
            if (data.auto_freq) {
              state.freq_min = data.freq_min;
              state.freq_max = data.freq_max;
              freqMinSlider.value = data.freq_min;
              freqMaxSlider.value = data.freq_max;
            }
            state.auto_speed = data.auto_speed;
            state.auto_freq = data.auto_freq;
            state.rhythm_division = data.rhythm_division;
            state.detected_bpm = data.detected_bpm;
            state.auto_gain = data.auto_gain;
            state.scale = data.scale;
            state.scale_type = data.scale_type;
            updateUI(state);
          }
        }
      } catch (err) {
        connDot.className = 'conn-dot';
      }
    }

    syncSliders();
    pollStatus();
    setInterval(pollStatus, 1200);
  </script>
</body>
</html>
)rawliteral";
}

#endif // RADAR_WEB_SERVER_H
