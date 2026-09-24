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
    std::atomic<float> decayTime{0.9700f};     // Decay persistence factor (default: Medium 0.9700)
    std::atomic<bool> autoDecay{true};         // Auto decay calibration mode (calibrated w.r.t. scanning speed)
    std::atomic<float> decaySpeedScale{1.0f};  // Decay speed multiplier (1x, 2x, 4x, etc.)
    std::atomic<float> freqMinHz{60.0f};       // Min frequency in Hz (default 60 Hz)
    std::atomic<float> freqMaxHz{4000.0f};     // Max frequency in Hz (default 4000 Hz)
    std::atomic<float> freqRangeMin{0.0f};     // Shader radial freq start (0.0 to 1.0)
    std::atomic<float> freqRangeMax{1.0f};     // Shader radial freq end (0.0 to 1.0)
    std::atomic<float> currentFps{0.0f};       // Measured FPS for live display
    std::atomic<int> frameCount{0};            // Total frames rendered

    // Frequency scaling: 0 = log, 1 = mel, 2 = linear
    std::atomic<int> scaleType{0};

    // Autotuning for scanning speed (musical rhythm sync) - Default ON
    std::atomic<bool> autoSpeed{true};
    std::atomic<int> rhythmDivision{2};       // 0: 1/4 note (1 beat), 1: 1/2 note, 2: 1 bar (default), 3: 2 bars, 4: 4 bars
    std::atomic<float> detectedBpm{120.0f};

    // Autotuning for frequency range / scaling
    std::atomic<bool> autoFreq{true};
    std::atomic<float> autoGain{1.0f};

    // NCA weight perturbation controls
    std::atomic<float> ncaPerturb{0.0f};      // 0.0 (unperturbed) to 1.0 (heavy perturbation)
    std::atomic<float> ncaSeed{0.0f};         // Mutation seed
    std::atomic<float> ncaStepScale{1.0f};    // NCA reaction step scale (default 1.0)

    // Topology control: 0.0 = Cartesian, 1.0 = Polar (pole in center) - Default: Polar Inward Rev
    std::atomic<float> ncaTopology{1.0f};      // Default: Polar (1.0)
    std::atomic<float> ncaPoleRadial{-1.0f};   // Default: Inward reverse pole (-1.0)
    std::atomic<float> ncaPoleAzimuthal{1.0f}; // +1.0 = CCW, -1.0 = CW
    std::atomic<bool> ncaReversePole{true};    // Default: true (Inward pole)
    std::atomic<bool> ncaReverseSpin{false};

    // Spectrogram influence & scanning line visibility (0.0 = completely muted / beam hidden)
    std::atomic<float> spectrogramInfluence{1.0f}; // Default: 1.0
    std::atomic<float> spectrogramScale{1.0f};     // Default: 1.0 (Radial scale multiplier)
    std::atomic<bool> resetStateRequested{false};

    // NCA Channel View (-1 = default composite, 12 = selectable 3 channels RGB, 13 = PCA RGB)
    std::atomic<int> ncaChannelView{-1};
    std::atomic<int> customChannelR{0};        // Default: Channel 0 for Red
    std::atomic<int> customChannelG{1};        // Default: Channel 1 for Green
    std::atomic<int> customChannelB{2};        // Default: Channel 2 for Blue

    // PCA Palette controls (re-roll orientation and hue shift)
    std::atomic<bool> pcaRerollRequested{false};
    std::atomic<float> pcaPendingHueShift{-999.0f};
    std::atomic<float> pcaPendingHueDelta{0.0f};
    std::atomic<float> pcaHueShift{0.0f};

    // Simulation Resolution configuration - Default: 800x600 (Relative 1:1)
    std::atomic<int> simWidth{800};
    std::atomic<int> simHeight{600};
    std::atomic<int> simScaleMode{0};          // 0 = Relative, 1 = Fixed
    std::atomic<float> simScale{1.0f};
    std::atomic<int> simFiltering{0};          // 0 = Smooth Hermite, 1 = Bilinear, 2 = Nearest
    std::atomic<float> ncaViewportDim{1.0f};
    std::atomic<bool> simResPending{false};
    std::mutex simResMutex;

    struct SimResolutionRequest {
        int width{800};
        int height{600};
        int scaleMode{0};
        float scale{1.0f};
        std::string preset{"1:1"};
        bool resample{true};
    };
    SimResolutionRequest pendingSimRes;
    std::string simPresetName{"1:1"};

    // Action triggers for Structure Imprint Easter Egg
    std::atomic<bool> hasPendingImprint{false};
    std::mutex imprintMutex;
    std::string pendingImprintMessage;

    void requestImprint(const std::string& msg) {
        std::lock_guard<std::mutex> lock(imprintMutex);
        pendingImprintMessage = msg;
        hasPendingImprint.store(true);
    }

    void requestSimulationResolution(int w, int h, int mode, float scale, const std::string& preset, bool resample = true) {
        std::lock_guard<std::mutex> lock(simResMutex);
        pendingSimRes.width = w;
        pendingSimRes.height = h;
        pendingSimRes.scaleMode = mode;
        pendingSimRes.scale = scale;
        pendingSimRes.preset = preset;
        pendingSimRes.resample = resample;
        simPresetName = preset;
        simWidth.store(w);
        simHeight.store(h);
        simScaleMode.store(mode);
        simScale.store(scale);
        simResPending.store(true);
    }

    bool hasPendingSimulationResolution() const {
        return simResPending.load();
    }

    bool popPendingSimulationResolution(SimResolutionRequest& req) {
        if (!simResPending.load()) return false;
        std::lock_guard<std::mutex> lock(simResMutex);
        if (!simResPending.load()) return false;
        req = pendingSimRes;
        simResPending.store(false);
        return true;
    }

    bool popPendingImprint(std::string& outMsg) {
        if (!hasPendingImprint.load()) return false;
        std::lock_guard<std::mutex> lock(imprintMutex);
        if (!hasPendingImprint.load()) return false;
        outMsg = pendingImprintMessage;
        hasPendingImprint.store(false);
        return true;
    }

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
        ss << "  \"auto_decay\": " << (autoDecay.load() ? "true" : "false") << ",\n";
        ss << "  \"decay_speed\": " << decaySpeedScale.load() << ",\n";
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
        ss << "  \"nca_perturb\": " << ncaPerturb.load() << ",\n";
        ss << "  \"nca_seed\": " << ncaSeed.load() << ",\n";
        ss << "  \"nca_step_scale\": " << ncaStepScale.load() << ",\n";
        ss << "  \"nca_topology\": " << ncaTopology.load() << ",\n";
        ss << "  \"topology\": \"" << (ncaTopology.load() > 0.5f ? "polar" : "cartesian") << "\",\n";
        ss << "  \"nca_pole_radial\": " << ncaPoleRadial.load() << ",\n";
        ss << "  \"nca_pole_azimuthal\": " << ncaPoleAzimuthal.load() << ",\n";
        ss << "  \"reverse_pole\": " << (ncaReversePole.load() ? "true" : "false") << ",\n";
        ss << "  \"reverse_spin\": " << (ncaReverseSpin.load() ? "true" : "false") << ",\n";
        ss << "  \"spectrogram_influence\": " << spectrogramInfluence.load() << ",\n";
        ss << "  \"spectrogram_scale\": " << spectrogramScale.load() << ",\n";
        ss << "  \"nca_channel_view\": " << ncaChannelView.load() << ",\n";
        ss << "  \"custom_channel_r\": " << customChannelR.load() << ",\n";
        ss << "  \"custom_channel_g\": " << customChannelG.load() << ",\n";
        ss << "  \"custom_channel_b\": " << customChannelB.load() << ",\n";
        ss << "  \"pca_hue_shift\": " << pcaHueShift.load() << ",\n";
        ss << "  \"sim_width\": " << simWidth.load() << ",\n";
        ss << "  \"sim_height\": " << simHeight.load() << ",\n";
        ss << "  \"sim_scale_mode\": " << simScaleMode.load() << ",\n";
        ss << "  \"sim_scale\": " << simScale.load() << ",\n";
        ss << "  \"sim_preset\": \"" << simPresetName << "\",\n";
        ss << "  \"sim_filtering\": " << simFiltering.load() << ",\n";
        ss << "  \"nca_viewport_dim\": " << ncaViewportDim.load() << ",\n";
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

        // Easter Egg Structure Imprint endpoint
        if (method == "POST" && cleanPath == "/api/imprint") {
            size_t bodyPos = request.find("\r\n\r\n");
            std::string body = (bodyPos != std::string::npos) ? request.substr(bodyPos + 4) : "";
            std::string msg = "BYOB";
            extractJsonString(body, "message", msg);
            m_config.requestImprint(msg);

            std::string resBody = "{\"status\":\"imprinted\",\"message\":\"" + msg + "\"}";
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

        // Simulation Resolution endpoint
        if (cleanPath == "/api/resolution" || cleanPath == "/api/genetic") {
            if (method == "POST") {
                size_t bodyPos = request.find("\r\n\r\n");
                std::string body = (bodyPos != std::string::npos) ? request.substr(bodyPos + 4) : "";
                
                int w = m_config.simWidth.load();
                int h = m_config.simHeight.load();
                int mode = m_config.simScaleMode.load();
                float scale = m_config.simScale.load();
                int filtering = m_config.simFiltering.load();
                bool resample = true;
                std::string preset = "";

                extractJsonString(body, "preset", preset);
                extractJsonInt(body, "width", w) || extractJsonInt(body, "w", w) || extractJsonInt(body, "sim_width", w);
                extractJsonInt(body, "height", h) || extractJsonInt(body, "h", h) || extractJsonInt(body, "sim_height", h);
                extractJsonFloat(body, "scale", scale) || extractJsonFloat(body, "sim_scale", scale);
                extractJsonInt(body, "scale_mode", mode) || extractJsonInt(body, "mode", mode) || extractJsonInt(body, "sim_scale_mode", mode);
                extractJsonInt(body, "filtering", filtering) || extractJsonInt(body, "sim_filtering", filtering);
                extractJsonBool(body, "resample", resample);

                if (filtering >= 0 && filtering <= 2) {
                    m_config.simFiltering.store(filtering);
                }

                if (!preset.empty()) {
                    if (preset == "native" || preset == "1:1") {
                        mode = 0; scale = 1.0f; w = 800; h = 600;
                    } else if (preset == "1:1.5" || preset == "2:3" || preset == "533x400") {
                        mode = 0; scale = 0.666667f; w = 533; h = 400;
                    } else if (preset == "half" || preset == "1:2" || preset == "400x300") {
                        mode = 0; scale = 0.5f; w = 400; h = 300;
                    } else if (preset == "original" || preset == "1:3" || preset == "default" || preset == "267x200") {
                        mode = 0; scale = 0.333333f; w = 267; h = 200;
                    } else if (preset == "quarter" || preset == "1:4" || preset == "200x150") {
                        mode = 0; scale = 0.25f; w = 200; h = 150;
                    } else if (preset == "sixth" || preset == "1:6" || preset == "133x100") {
                        mode = 0; scale = 0.166667f; w = 133; h = 100;
                    } else if (preset == "160x120") {
                        mode = 1; w = 160; h = 120;
                    } else if (preset == "200x150") {
                        mode = 1; w = 200; h = 150;
                    } else if (preset == "320x240") {
                        mode = 1; w = 320; h = 240;
                    } else if (preset == "400x300") {
                        mode = 1; w = 400; h = 300;
                    } else if (preset == "512x384") {
                        mode = 1; w = 512; h = 384;
                    } else if (preset == "640x480") {
                        mode = 1; w = 640; h = 480;
                    } else if (preset == "800x600") {
                        mode = 1; w = 800; h = 600;
                    }
                } else if (scale > 0.05f && scale <= 3.0f && (w == m_config.simWidth.load() && h == m_config.simHeight.load())) {
                    w = std::max(32, (int)std::round(800.0f * scale));
                    h = std::max(24, (int)std::round(600.0f * scale));
                    mode = 0;
                    std::ostringstream ssPreset;
                    ssPreset << std::fixed << std::setprecision(2) << scale << "x";
                    preset = ssPreset.str();
                }

                if (preset.empty()) {
                    preset = std::to_string(w) + "x" + std::to_string(h);
                }

                m_config.simScaleMode.store(mode);
                m_config.simScale.store(scale);
                m_config.requestSimulationResolution(w, h, mode, scale, preset, resample);
                for (int waitIter = 0; waitIter < 100; waitIter++) {
                    if (!m_config.hasPendingSimulationResolution() && m_config.simWidth.load() == w) break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
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

        // Reset Simulation State endpoint
        if (cleanPath == "/api/reset_state") {
            if (method == "POST") {
                m_config.resetStateRequested.store(true);
                std::this_thread::sleep_for(std::chrono::milliseconds(25));
            }
            std::string resBody = "{\"status\":\"state_reset\"}";
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

        // Spectrogram influence endpoint
        if (cleanPath == "/api/spectrogram") {
            if (method == "POST") {
                size_t bodyPos = request.find("\r\n\r\n");
                std::string body = (bodyPos != std::string::npos) ? request.substr(bodyPos + 4) : "";
                float val;
                if (extractJsonFloat(body, "influence", val) || extractJsonFloat(body, "spectrogram_influence", val) || extractJsonFloat(body, "gain", val)) {
                    m_config.spectrogramInfluence.store(std::max(0.0f, std::min(5.0f, val)));
                }
                if (extractJsonFloat(body, "scale", val) || extractJsonFloat(body, "spectrogram_scale", val) || extractJsonFloat(body, "multiplier", val)) {
                    m_config.spectrogramScale.store(std::max(0.1f, std::min(10.0f, val)));
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

        // PCA Palette control endpoint (re-roll orientation and hue shift)
        if (cleanPath == "/api/pca/palette" || cleanPath == "/api/palette") {
            if (method == "POST") {
                size_t bodyPos = request.find("\r\n\r\n");
                std::string body = (bodyPos != std::string::npos) ? request.substr(bodyPos + 4) : "";
                std::string action;
                bool reroll = false;
                if ((extractJsonBool(body, "reroll", reroll) && reroll) || (extractJsonString(body, "action", action) && action == "reroll")) {
                    m_config.pcaRerollRequested.store(true);
                }
                float shift;
                if (extractJsonFloat(body, "shift", shift) || extractJsonFloat(body, "delta", shift) || extractJsonFloat(body, "hue_delta", shift)) {
                    if (std::abs(shift) <= 6.2832f && std::abs(shift) > 0.0001f) {
                        shift = shift * 180.0f / (float)M_PI;
                    }
                    m_config.pcaPendingHueDelta.store(shift);
                }
                float hue;
                if (extractJsonFloat(body, "hue", hue) || extractJsonFloat(body, "hue_shift", hue) || extractJsonFloat(body, "pca_hue_shift", hue)) {
                    if (hue <= 6.2832f && hue > 0.0001f) {
                        hue = hue * 180.0f / (float)M_PI;
                    }
                    m_config.pcaPendingHueShift.store(hue);
                    m_config.pcaHueShift.store(hue);
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

        // Direct NCA Channel View & Custom 3-Channel selection
        if (cleanPath == "/api/channel" || cleanPath == "/api/channels") {
            if (method == "POST") {
                size_t bodyPos = request.find("\r\n\r\n");
                std::string body = (bodyPos != std::string::npos) ? request.substr(bodyPos + 4) : "";
                int chVal;
                std::string chStr;
                bool hasExplicitChannel = false;
                if (extractJsonInt(body, "channel", chVal) || extractJsonInt(body, "nca_channel_view", chVal)) {
                    if (chVal >= -1 && chVal <= 13) {
                        m_config.ncaChannelView.store(chVal);
                        hasExplicitChannel = true;
                    }
                } else if (extractJsonString(body, "channel", chStr) || extractJsonString(body, "mode", chStr)) {
                    if (chStr == "pca" || chStr == "pca_rgb" || chStr == "pca3") {
                        m_config.ncaChannelView.store(13);
                        hasExplicitChannel = true;
                    } else if (chStr == "composite" || chStr == "default") {
                        m_config.ncaChannelView.store(-1);
                        hasExplicitChannel = true;
                    } else if (chStr == "custom" || chStr == "3ch" || chStr == "rgb" || chStr == "raw_rgb") {
                        m_config.ncaChannelView.store(12);
                        hasExplicitChannel = true;
                    }
                }
                int r, g, b;
                bool customChannelsSent = false;
                if (extractJsonInt(body, "custom_r", r) || extractJsonInt(body, "r", r) || extractJsonInt(body, "custom_channel_r", r)) {
                    m_config.customChannelR.store(std::max(0, std::min(11, r)));
                    customChannelsSent = true;
                }
                if (extractJsonInt(body, "custom_g", g) || extractJsonInt(body, "g", g) || extractJsonInt(body, "custom_channel_g", g)) {
                    m_config.customChannelG.store(std::max(0, std::min(11, g)));
                    customChannelsSent = true;
                }
                if (extractJsonInt(body, "custom_b", b) || extractJsonInt(body, "b", b) || extractJsonInt(body, "custom_channel_b", b)) {
                    m_config.customChannelB.store(std::max(0, std::min(11, b)));
                    customChannelsSent = true;
                }
                if (!hasExplicitChannel && customChannelsSent) {
                    m_config.ncaChannelView.store(12);
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

        // Decay persistence & speed calibration endpoint
        if (cleanPath == "/api/decay") {
            if (method == "POST") {
                size_t bodyPos = request.find("\r\n\r\n");
                std::string body = (bodyPos != std::string::npos) ? request.substr(bodyPos + 4) : "";
                bool boolVal;
                if (extractJsonBool(body, "auto_decay", boolVal) || extractJsonBool(body, "auto", boolVal)) {
                    m_config.autoDecay.store(boolVal);
                }
                float val;
                if (extractJsonFloat(body, "decay_speed", val) || extractJsonFloat(body, "speed", val) || extractJsonFloat(body, "decay_scale", val) || extractJsonFloat(body, "multiplier", val)) {
                    m_config.decaySpeedScale.store(std::max(0.1f, std::min(10.0f, val)));
                }
                if (extractJsonFloat(body, "decay_time", val) || extractJsonFloat(body, "decay", val)) {
                    m_config.decayTime.store(std::max(0.1f, std::min(0.9999f, val)));
                    m_config.autoDecay.store(false);
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

        // NCA Topology endpoint
        if (cleanPath == "/api/topology") {
            if (method == "POST") {
                size_t bodyPos = request.find("\r\n\r\n");
                std::string body = (bodyPos != std::string::npos) ? request.substr(bodyPos + 4) : "";
                float topVal = 0.0f;
                std::string topStr;
                if (extractJsonFloat(body, "nca_topology", topVal)) {
                    m_config.ncaTopology.store(std::max(0.0f, std::min(1.0f, topVal)));
                } else if (extractJsonString(body, "topology", topStr) || extractJsonString(body, "mode", topStr)) {
                    if (topStr == "polar") m_config.ncaTopology.store(1.0f);
                    else if (topStr == "cartesian") m_config.ncaTopology.store(0.0f);
                    else if (topStr == "spiral" || topStr == "vortex") m_config.ncaTopology.store(0.5f);
                }

                bool revPole = false;
                if (extractJsonBool(body, "reverse_pole", revPole)) {
                    m_config.ncaReversePole.store(revPole);
                    m_config.ncaPoleRadial.store(revPole ? -1.0f : 1.0f);
                }
                bool revSpin = false;
                if (extractJsonBool(body, "reverse_spin", revSpin)) {
                    m_config.ncaReverseSpin.store(revSpin);
                    m_config.ncaPoleAzimuthal.store(revSpin ? -1.0f : 1.0f);
                }
                float radVal;
                if (extractJsonFloat(body, "nca_pole_radial", radVal)) {
                    m_config.ncaPoleRadial.store(radVal);
                    m_config.ncaReversePole.store(radVal < 0.0f);
                }
                float azimVal;
                if (extractJsonFloat(body, "nca_pole_azimuthal", azimVal)) {
                    m_config.ncaPoleAzimuthal.store(azimVal);
                    m_config.ncaReverseSpin.store(azimVal < 0.0f);
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

        // Re-roll settings endpoint (kept for backwards compatibility)
        if (cleanPath == "/api/reroll") {
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
            if (extractJsonBool(body, "auto_decay", boolVal)) {
                m_config.autoDecay.store(boolVal);
            }
            if (extractJsonFloat(body, "decay_speed", val) || extractJsonFloat(body, "decay_speed_scale", val) || extractJsonFloat(body, "decay_scale", val) || extractJsonFloat(body, "decay_speed_mult", val)) {
                m_config.decaySpeedScale.store(std::max(0.1f, std::min(10.0f, val)));
            }
            if (extractJsonFloat(body, "decay_multiply", val) && val > 0.01f) {
                float curScale = m_config.decaySpeedScale.load();
                m_config.decaySpeedScale.store(std::max(0.1f, std::min(10.0f, curScale * val)));
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

            // Speed multiplier (e.g. 0.25x, 0.5x, 2.0x, 4.0x)
            if (extractJsonFloat(body, "speed_multiplier", val)) {
                if (val > 0.001f && val <= 10.0f) {
                    float cur = m_config.scanningSpeed.load();
                    float newSpeed = std::max(0.01f, std::min(10.0f, cur * val));
                    m_config.scanningSpeed.store(newSpeed);
                    m_config.autoSpeed.store(false);
                }
            }

            // NCA weight perturbation controls
            if (extractJsonFloat(body, "nca_perturb", val)) {
                if (val >= 0.0f && val <= 2.0f) {
                    m_config.ncaPerturb.store(val);
                }
            }
            if (extractJsonFloat(body, "nca_seed", val)) {
                m_config.ncaSeed.store(val);
            }
            if (extractJsonFloat(body, "nca_step_scale", val)) {
                if (val >= 0.05f && val <= 5.0f) {
                    m_config.ncaStepScale.store(val);
                }
            }
            if (extractJsonFloat(body, "nca_topology", val)) {
                m_config.ncaTopology.store(std::max(0.0f, std::min(1.0f, val)));
            } else if (extractJsonString(body, "topology", strVal)) {
                if (strVal == "polar") m_config.ncaTopology.store(1.0f);
                else if (strVal == "cartesian") m_config.ncaTopology.store(0.0f);
                else if (strVal == "spiral" || strVal == "vortex") m_config.ncaTopology.store(0.5f);
            }
            if (extractJsonBool(body, "reverse_pole", boolVal)) {
                m_config.ncaReversePole.store(boolVal);
                m_config.ncaPoleRadial.store(boolVal ? -1.0f : 1.0f);
            }
            if (extractJsonBool(body, "reverse_spin", boolVal)) {
                m_config.ncaReverseSpin.store(boolVal);
                m_config.ncaPoleAzimuthal.store(boolVal ? -1.0f : 1.0f);
            }
            if (extractJsonFloat(body, "nca_pole_radial", val)) {
                m_config.ncaPoleRadial.store(val);
                m_config.ncaReversePole.store(val < 0.0f);
            }
            if (extractJsonFloat(body, "nca_pole_azimuthal", val)) {
                m_config.ncaPoleAzimuthal.store(val);
                m_config.ncaReverseSpin.store(val < 0.0f);
            }
            std::string presetStr;
            if (extractJsonString(body, "sim_preset", presetStr) || extractJsonString(body, "preset", presetStr)) {
                int w = m_config.simWidth.load();
                int h = m_config.simHeight.load();
                int mode = m_config.simScaleMode.load();
                float scale = m_config.simScale.load();
                if (presetStr == "native" || presetStr == "1:1") {
                    mode = 0; scale = 1.0f; w = 800; h = 600;
                } else if (presetStr == "1:1.5" || presetStr == "2:3") {
                    mode = 0; scale = 0.666667f; w = 533; h = 400;
                } else if (presetStr == "half" || presetStr == "1:2") {
                    mode = 0; scale = 0.5f; w = 400; h = 300;
                } else if (presetStr == "original" || presetStr == "1:3" || presetStr == "default") {
                    mode = 0; scale = 0.333333f; w = 267; h = 200;
                } else if (presetStr == "quarter" || presetStr == "1:4") {
                    mode = 0; scale = 0.25f; w = 200; h = 150;
                } else if (presetStr == "sixth" || presetStr == "1:6") {
                    mode = 0; scale = 0.166667f; w = 133; h = 100;
                } else if (presetStr == "160x120") {
                    mode = 1; w = 160; h = 120;
                } else if (presetStr == "200x150") {
                    mode = 1; w = 200; h = 150;
                } else if (presetStr == "267x200") {
                    mode = 1; w = 267; h = 200;
                } else if (presetStr == "320x240") {
                    mode = 1; w = 320; h = 240;
                } else if (presetStr == "400x300") {
                    mode = 1; w = 400; h = 300;
                } else if (presetStr == "512x384") {
                    mode = 1; w = 512; h = 384;
                } else if (presetStr == "640x480") {
                    mode = 1; w = 640; h = 480;
                } else if (presetStr == "800x600") {
                    mode = 1; w = 800; h = 600;
                }
                m_config.simScaleMode.store(mode);
                m_config.simScale.store(scale);
                m_config.requestSimulationResolution(w, h, mode, scale, presetStr, true);
                for (int waitIter = 0; waitIter < 100; waitIter++) {
                    if (!m_config.hasPendingSimulationResolution() && m_config.simWidth.load() == w) break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
            }
            if (extractJsonFloat(body, "sim_scale", val)) {
                if (val >= 0.05f && val <= 3.0f) {
                    int w = std::max(32, (int)std::round(800.0f * val));
                    int h = std::max(24, (int)std::round(600.0f * val));
                    std::ostringstream ssPreset;
                    ssPreset << std::fixed << std::setprecision(2) << val << "x";
                    m_config.simScaleMode.store(0);
                    m_config.simScale.store(val);
                    m_config.requestSimulationResolution(w, h, 0, val, ssPreset.str(), true);
                    for (int waitIter = 0; waitIter < 30; waitIter++) {
                        if (!m_config.hasPendingSimulationResolution() && m_config.simWidth.load() == w) break;
                        std::this_thread::sleep_for(std::chrono::milliseconds(5));
                    }
                }
            }
            if (extractJsonInt(body, "sim_filtering", intVal)) {
                if (intVal >= 0 && intVal <= 2) {
                    m_config.simFiltering.store(intVal);
                }
            }

            if (extractJsonFloat(body, "spectrogram_influence", val) || extractJsonFloat(body, "spectrogram_gain", val) || extractJsonFloat(body, "spectrogram_strength", val)) {
                m_config.spectrogramInfluence.store(std::max(0.0f, std::min(5.0f, val)));
            }
            if (extractJsonFloat(body, "spectrogram_scale", val) || extractJsonFloat(body, "spec_scale", val)) {
                m_config.spectrogramScale.store(std::max(0.1f, std::min(10.0f, val)));
            }
            if ((extractJsonBool(body, "reset_state", boolVal) && boolVal) || (extractJsonString(body, "action", strVal) && strVal == "reset_state")) {
                m_config.resetStateRequested.store(true);
            }
            if (extractJsonString(body, "imprint_message", strVal)) {
                m_config.requestImprint(strVal);
            }
            bool hasExplicitChannelView = false;
            if (extractJsonInt(body, "nca_channel_view", intVal) || extractJsonInt(body, "channel", intVal)) {
                if (intVal >= -1 && intVal <= 13) {
                    m_config.ncaChannelView.store(intVal);
                    hasExplicitChannelView = true;
                }
            }
            int chVal;
            bool customChannelsSent = false;
            if (extractJsonInt(body, "custom_channel_r", chVal) || extractJsonInt(body, "custom_r", chVal) || extractJsonInt(body, "r", chVal)) {
                m_config.customChannelR.store(std::max(0, std::min(11, chVal)));
                customChannelsSent = true;
            }
            if (extractJsonInt(body, "custom_channel_g", chVal) || extractJsonInt(body, "custom_g", chVal) || extractJsonInt(body, "g", chVal)) {
                m_config.customChannelG.store(std::max(0, std::min(11, chVal)));
                customChannelsSent = true;
            }
            if (extractJsonInt(body, "custom_channel_b", chVal) || extractJsonInt(body, "custom_b", chVal) || extractJsonInt(body, "b", chVal)) {
                m_config.customChannelB.store(std::max(0, std::min(11, chVal)));
                customChannelsSent = true;
            }
            if (!hasExplicitChannelView && customChannelsSent) {
                m_config.ncaChannelView.store(12);
            }
            if (extractJsonBool(body, "pca_reroll", boolVal) && boolVal) {
                m_config.pcaRerollRequested.store(true);
            }
            if (extractJsonFloat(body, "pca_hue_shift", val) || extractJsonFloat(body, "pca_shift", val)) {
                if (std::abs(val - m_config.pcaHueShift.load()) > 0.001f) {
                    m_config.pcaPendingHueShift.store(val);
                    m_config.pcaHueShift.store(val);
                }
            }
            if (extractJsonFloat(body, "pca_hue_delta", val)) {
                if (std::abs(val) > 0.001f) {
                    m_config.pcaPendingHueDelta.store(val);
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
      --accent-cyan: #00e5ff;
      --accent-warn: #ffbb00;
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
      padding: 20px 16px;
    }
    .panel {
      width: 100%;
      max-width: 520px;
      display: flex;
      flex-direction: column;
      gap: 14px;
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
      padding: 14px 16px;
      display: flex;
      flex-direction: column;
      gap: 9px;
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
    input[type=text] {
      background: var(--track);
      border: 1px solid var(--card-border);
      color: var(--text);
      padding: 7px 10px;
      border-radius: 5px;
      font-family: monospace;
      font-size: 0.85rem;
      outline: none;
      transition: border-color 0.2s;
    }
    input[type=text]:focus {
      border-color: var(--accent);
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
      padding: 6px 8px;
      border-radius: 5px;
      font-size: 0.73rem;
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
    button.toggle-btn:hover {
      border-color: var(--accent);
      color: var(--accent);
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
    .badge {
      font-size: 0.70rem;
      font-weight: 700;
      padding: 2px 7px;
      border-radius: 4px;
      text-transform: uppercase;
      letter-spacing: 0.5px;
    }
    .badge-stable {
      background: #15222d;
      color: var(--text-muted);
      border: 1px solid var(--card-border);
    }
    .badge-candidate {
      background: rgba(255, 187, 0, 0.15);
      color: var(--accent-warn);
      border: 1px solid var(--accent-warn);
      box-shadow: 0 0 8px rgba(255, 187, 0, 0.3);
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
      
      <!-- Speed multiplier controls: 2x/4x faster / slower -->
      <div class="btn-row">
        <button class="btn" onclick="multiplySpeed(0.25)" title="4x Slower">¼x Slower</button>
        <button class="btn" onclick="multiplySpeed(0.5)" title="2x Slower">½x Slower</button>
        <button class="btn" onclick="multiplySpeed(2.0)" title="2x Faster">2x Faster</button>
        <button class="btn" onclick="multiplySpeed(4.0)" title="4x Faster">4x Faster</button>
      </div>

      <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; margin-top:2px;">
        <button id="autoSpeedBtn" class="toggle-btn active" onclick="toggleAutoSpeed()">
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

    <!-- Spectrogram Influence & Scanning Beam Control -->
    <div class="section">
      <div class="section-head">
        <span class="section-title">Spectrogram Influence</span>
        <span id="specInfluenceVal" class="val">1.00</span>
      </div>
      <input type="range" id="specInfluenceSlider" min="0.00" max="2.00" step="0.05" value="1.00">
      <div class="btn-row">
        <button class="btn" id="specBtn0" onclick="setSpectrogramInfluence(0.0)" title="Spectrogram does not affect state; scanning beam hidden">0.0 (Hide Line)</button>
        <button class="btn" id="specBtn50" onclick="setSpectrogramInfluence(0.5)">0.5 (Subtle)</button>
        <button class="btn active" id="specBtn100" onclick="setSpectrogramInfluence(1.0)">1.0 (Default)</button>
        <button class="btn" id="specBtn150" onclick="setSpectrogramInfluence(1.5)">1.5 (Strong)</button>
      </div>
      <div class="meta-row">
        <span>Audio State Coupling:</span>
        <span id="specInfluenceDesc">Active Scanning Beam</span>
      </div>
    </div>

    <!-- NCA Topology: Cartesian vs Polar (Rotated Sobel Filters) with Reverse Pole options -->
    <div class="section">
      <div class="section-head">
        <span class="section-title">NCA Topology</span>
        <span id="topologyLabel" class="val">Polar (Inward Pole)</span>
      </div>
      <div class="btn-row">
        <button class="btn" id="topoBtnCart" onclick="setTopology(0.0)">Cartesian</button>
        <button class="btn" id="topoBtnSpiral" onclick="setTopology(0.5)">Spiral / Vortex</button>
        <button class="btn active" id="topoBtnPolar" onclick="setTopology(1.0)">Polar (Center)</button>
      </div>
      <input type="range" id="topologySlider" min="0.0" max="1.0" step="0.02" value="1.0">
      
      <!-- Reverse Pole options -->
      <div class="btn-row" style="margin-top:2px;">
        <button class="toggle-btn active" id="reversePoleBtn" style="flex:1;" onclick="toggleReversePole()" title="Reverse radial pole direction (inward vs outward)">
          🔄 Reverse Pole: Inward
        </button>
        <button class="toggle-btn" id="reverseSpinBtn" style="flex:1;" onclick="toggleReverseSpin()" title="Reverse azimuthal rotation (CW vs CCW)">
          🔃 Spin: CCW
        </button>
      </div>

      <div class="meta-row">
        <span>Sobel Filter Field:</span>
        <span id="topologyDesc">Inward Sink ∇r, Azimuthal ∇θ</span>
      </div>
    </div>

    <!-- NCA Channels (Direct View, 3 Selectable Channels & PCA Palette Controls) -->
    <div class="section">
      <div class="section-head">
        <span class="section-title">NCA Channels</span>
        <span id="channelViewLabel" class="val">Composite (CC)</span>
      </div>

      <!-- Channel Selector Modes -->
      <div class="btn-row">
        <button class="btn active" id="chBtnComp" onclick="setNcaChannel(-1)" title="Standard composite output with purple/green color correction">Composite (CC)</button>
        <button class="btn" id="chBtnCustom" onclick="setNcaChannel(12)" title="Display 3 user-selected channels directly mapped to RGB w/o color correction">3 Channels (RGB)</button>
        <button class="btn" id="chBtnPca" onclick="setNcaChannel(13)" title="3 main PCA components projected to RGB (w/o color correction)" style="border-color:#bb66ff; color:#dd88ff; font-weight:700;">🌈 PCA RGB (3 PCs)</button>
      </div>

      <!-- 3 Selectable Channels Selector Bar -->
      <div id="customChannelsPanel" style="display:flex; flex-direction:column; gap:5px; background:rgba(0,0,0,0.25); border:1px solid var(--card-border); border-radius:5px; padding:6px 10px; margin-top:2px;">
        <div class="meta-row" style="margin:0;">
          <span style="font-weight:600; color:var(--text-muted);">Custom 3-Channel RGB Mapping:</span>
          <span id="customChannelsDesc" style="color:var(--accent); font-weight:700;">R: Ch 0 | G: Ch 1 | B: Ch 2</span>
        </div>
        <div style="display:flex; gap:8px; align-items:center;">
          <div style="flex:1; display:flex; flex-direction:column; gap:2px;">
            <label style="font-size:0.68rem; color:#ff7777; font-weight:700;">Red Channel</label>
            <select id="selChR" onchange="updateCustomChannels()" style="background:#15222d; color:#ffffff; border:1px solid #ff5555; border-radius:4px; padding:4px; font-size:0.75rem; font-family:monospace; cursor:pointer;">
              <option value="0" selected>Ch 0</option>
              <option value="1">Ch 1</option>
              <option value="2">Ch 2</option>
              <option value="3">Ch 3</option>
              <option value="4">Ch 4</option>
              <option value="5">Ch 5</option>
              <option value="6">Ch 6</option>
              <option value="7">Ch 7</option>
              <option value="8">Ch 8</option>
              <option value="9">Ch 9</option>
              <option value="10">Ch 10</option>
              <option value="11">Ch 11</option>
            </select>
          </div>
          <div style="flex:1; display:flex; flex-direction:column; gap:2px;">
            <label style="font-size:0.68rem; color:#55ff88; font-weight:700;">Green Channel</label>
            <select id="selChG" onchange="updateCustomChannels()" style="background:#15222d; color:#ffffff; border:1px solid #33cc66; border-radius:4px; padding:4px; font-size:0.75rem; font-family:monospace; cursor:pointer;">
              <option value="0">Ch 0</option>
              <option value="1" selected>Ch 1</option>
              <option value="2">Ch 2</option>
              <option value="3">Ch 3</option>
              <option value="4">Ch 4</option>
              <option value="5">Ch 5</option>
              <option value="6">Ch 6</option>
              <option value="7">Ch 7</option>
              <option value="8">Ch 8</option>
              <option value="9">Ch 9</option>
              <option value="10">Ch 10</option>
              <option value="11">Ch 11</option>
            </select>
          </div>
          <div style="flex:1; display:flex; flex-direction:column; gap:2px;">
            <label style="font-size:0.68rem; color:#66aaff; font-weight:700;">Blue Channel</label>
            <select id="selChB" onchange="updateCustomChannels()" style="background:#15222d; color:#ffffff; border:1px solid #4488ff; border-radius:4px; padding:4px; font-size:0.75rem; font-family:monospace; cursor:pointer;">
              <option value="0">Ch 0</option>
              <option value="1">Ch 1</option>
              <option value="2" selected>Ch 2</option>
              <option value="3">Ch 3</option>
              <option value="4">Ch 4</option>
              <option value="5">Ch 5</option>
              <option value="6">Ch 6</option>
              <option value="7">Ch 7</option>
              <option value="8">Ch 8</option>
              <option value="9">Ch 9</option>
              <option value="10">Ch 10</option>
              <option value="11">Ch 11</option>
            </select>
          </div>
        </div>
      </div>

      <!-- PCA Palette Controls (Re-roll & Shift Palette) -->
      <div id="pcaPalettePanel" style="display:flex; flex-direction:column; gap:5px; background:rgba(187,102,255,0.08); border:1px solid rgba(187,102,255,0.3); border-radius:5px; padding:6px 10px; margin-top:2px;">
        <div class="meta-row" style="margin:0;">
          <span style="font-weight:600; color:#dd88ff;">PCA Palette Controls:</span>
          <span id="pcaHueVal" style="color:var(--accent-cyan); font-family:monospace;">0° (Original)</span>
        </div>
        <div class="btn-row" style="margin:0;">
          <button class="btn" onclick="rerollPcaPalette()" style="flex:1.2; border-color:#bb66ff; color:#dd88ff; font-weight:700;" title="Re-roll PCA color axes randomly in SO(3)">🎲 Re-roll Palette</button>
          <button class="btn" onclick="shiftPcaPalette(45)" style="flex:1;" title="Shift palette hue by +45° around achromatic axis">🔄 Shift +45°</button>
          <button class="btn" onclick="shiftPcaPalette(90)" style="flex:1;" title="Shift palette hue by +90° around achromatic axis">🔄 Shift +90°</button>
        </div>
        <div style="display:flex; flex-direction:column; gap:2px; margin-top:2px;">
          <div class="meta-row" style="margin:0;">
            <span style="font-size:0.70rem;">Continuous Hue Rotation</span>
            <span id="pcaHueAngleLabel" style="font-size:0.70rem; font-family:monospace;">0°</span>
          </div>
          <input type="range" id="pcaHueSlider" min="0" max="360" step="1" value="0" oninput="onPcaHueSlider(this.value)">
        </div>
      </div>

      <div class="meta-row" style="margin-top:4px;">
        <span>Direct Grayscale Channel View:</span>
      </div>
      <div class="btn-row">
        <button class="btn" id="chBtn0" onclick="setNcaChannel(0)" title="Channel 0 (Direct raw magnitude)">Ch 0</button>
        <button class="btn" id="chBtn1" onclick="setNcaChannel(1)" title="Channel 1 (Direct raw magnitude)">Ch 1</button>
        <button class="btn" id="chBtn2" onclick="setNcaChannel(2)" title="Channel 2 (Direct raw magnitude)">Ch 2</button>
        <button class="btn" id="chBtn3" onclick="setNcaChannel(3)" title="Channel 3 (Direct raw magnitude)">Ch 3</button>
      </div>
      <div class="btn-row">
        <button class="btn" id="chBtn4" onclick="setNcaChannel(4)" title="Channel 4 (Direct raw magnitude)">Ch 4</button>
        <button class="btn" id="chBtn5" onclick="setNcaChannel(5)" title="Channel 5 (Direct raw magnitude)">Ch 5</button>
        <button class="btn" id="chBtn6" onclick="setNcaChannel(6)" title="Channel 6 (Direct raw magnitude)">Ch 6</button>
        <button class="btn" id="chBtn7" onclick="setNcaChannel(7)" title="Channel 7 (Direct raw magnitude)">Ch 7</button>
      </div>
      <div class="btn-row">
        <button class="btn" id="chBtn8" onclick="setNcaChannel(8)" title="Channel 8 (Direct raw magnitude)">Ch 8</button>
        <button class="btn" id="chBtn9" onclick="setNcaChannel(9)" title="Channel 9 (Direct raw magnitude)">Ch 9</button>
        <button class="btn" id="chBtn10" onclick="setNcaChannel(10)" title="Channel 10 (Direct raw magnitude)">Ch 10</button>
        <button class="btn" id="chBtn11" onclick="setNcaChannel(11)" title="Channel 11 (Direct raw magnitude)">Ch 11</button>
      </div>

      <div class="meta-row">
        <span>Direct View:</span>
        <span id="channelViewDesc">Color corrected composite (Purple/Green)</span>
      </div>
    </div>

    <!-- Decay Time -->
    <!-- Decay Persistence & Speed Control -->
    <div class="section">
      <div class="section-head">
        <span class="section-title">Decay Persistence & Speed</span>
        <span id="decayVal" class="val">0.9700</span>
      </div>

      <!-- Decay Speed Controls: 2x / 4x Buttons -->
      <div style="display:flex; flex-direction:column; gap:3px;">
        <div class="meta-row">
          <span>Decay Speed:</span>
          <span id="decaySpeedVal" class="val" style="color:var(--accent-cyan); font-weight:700;">1.00x</span>
        </div>
        <div class="btn-row">
          <button class="btn" id="decaySpeedBtn05" onclick="setDecaySpeed(0.5)" title="0.5x Slower Decay (Longer persistence)">0.5x</button>
          <button class="btn active" id="decaySpeedBtn1" onclick="setDecaySpeed(1.0)" title="1x Normal Decay Speed">1x</button>
          <button class="btn" id="decaySpeedBtn2" onclick="setDecaySpeed(2.0)" title="2x Faster Decay Speed">2x</button>
          <button class="btn" id="decaySpeedBtn4" onclick="setDecaySpeed(4.0)" title="4x Faster Decay Speed">4x</button>
        </div>
      </div>

      <div class="meta-row" style="margin-top:4px;">
        <span>Radar Trail Persistence:</span>
        <span id="decayLabel" class="val" style="color:var(--accent); font-weight:700;">Medium (0.9700)</span>
      </div>
      <div class="btn-row">
        <button class="btn" id="decayFastBtn" onclick="setDecay(0.9200)">Fast</button>
        <button class="btn active" id="decayMedBtn" onclick="setDecay(0.9700)">Medium</button>
        <button class="btn" id="decayDefBtn" onclick="setDecay(0.9950)">Default</button>
        <button class="btn" id="decayLongBtn" onclick="setDecay(0.9990)">Long Glow</button>
      </div>
      <input type="range" id="decaySlider" min="0.8000" max="0.9995" step="0.0005" value="0.9700">

      <div style="display:flex; justify-content:space-between; align-items:center; margin-top:4px;">
        <button id="autoDecayBtn" class="toggle-btn active" onclick="toggleAutoDecay()" title="Automatically calibrate decay persistence to scanning speed so perturbation decays consistently w.r.t. radar rotation">
          ⚡ Auto Decay Calibrate
        </button>
        <span id="autoDecayStatus" class="meta-row" style="margin:0; font-size:0.75rem;">Sync to Sweep</span>
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
        <span class="section-title">Frequency Range & Spectrogram Scale</span>
        <span id="freqRangeVal" class="val">60 - 4000 Hz</span>
      </div>

      <!-- Spectrogram Scale Multipliers: 2x/4x bigger w/o disabling auto range (like rhythm section) -->
      <div style="display:flex; flex-direction:column; gap:3px;">
        <div class="meta-row">
          <span>Spectrogram Scale (Size):</span>
          <span id="specScaleVal" class="val" style="color:var(--accent); font-weight:700;">1.00x</span>
        </div>
        <div class="btn-row">
          <button class="btn" id="specScaleBtn05" onclick="setSpectrogramScale(0.5)" title="½x Smaller Spectrogram">½x Smaller</button>
          <button class="btn active" id="specScaleBtn1" onclick="setSpectrogramScale(1.0)" title="1x Normal Spectrogram Size">1x Normal</button>
          <button class="btn" id="specScaleBtn2" onclick="setSpectrogramScale(2.0)" title="2x Bigger Spectrogram (Magnified)">2x Bigger</button>
          <button class="btn" id="specScaleBtn4" onclick="setSpectrogramScale(4.0)" title="4x Bigger Spectrogram (Super Magnified)">4x Bigger</button>
        </div>
        <input type="range" id="specScaleSlider" min="0.25" max="5.00" step="0.05" value="1.00">
      </div>

      <div style="display:flex; flex-direction:column; gap:3px; margin-top:4px;">
        <div class="meta-row"><span>Min Freq</span><span id="minFreqVal">60 Hz</span></div>
        <input type="range" id="freqMinSlider" min="20" max="2000" step="10" value="60">
      </div>

      <div style="display:flex; flex-direction:column; gap:3px; margin-top:2px;">
        <div class="meta-row"><span>Max Freq</span><span id="maxFreqVal">4000 Hz</span></div>
        <input type="range" id="freqMaxSlider" min="200" max="20000" step="100" value="4000">
      </div>

      <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; margin-top:2px;">
        <button id="autoFreqBtn" class="toggle-btn active" onclick="toggleAutoFreq()">
          ⚡ Auto Freq & Scaling
        </button>
        <span id="gainLabel" class="meta-row" style="margin:0;">Gain: 1.0x</span>
      </div>
    </div>

    <!-- Simulation Resolution & State Card -->
    <div class="section">
      <div class="section-head">
        <span class="section-title">Simulation Resolution</span>
        <span id="simResBadge" class="val">800 × 600 (1:1)</span>
      </div>

      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span id="simInfoLabel" class="meta-row" style="margin:0;">480.0k cells &bull; Relative 1:1</span>
        <span id="dimStatusLabel" class="meta-row" style="margin:0; color:var(--accent-cyan); font-weight:600;">Active (100% Energy)</span>
      </div>

      <!-- Relative Scale Presets (Screen Proportional) -->
      <div style="display:flex; flex-direction:column; gap:3px; margin-top:2px;">
        <div class="meta-row"><span>Relative Scale Presets (Screen Proportional)</span></div>
        <div class="btn-row" style="flex-wrap:wrap;">
          <button class="btn active" id="presetBtn1_1" onclick="setSimPreset('1:1')" title="1:1 Full Native Screen Resolution">1:1 Native</button>
          <button class="btn" id="presetBtn2_3" onclick="setSimPreset('1:1.5')" title="2:3 Scale (~533x400)">2:3 (1:1.5)</button>
          <button class="btn" id="presetBtn1_2" onclick="setSimPreset('1:2')" title="1:2 Half Resolution (400x300)">1:2 Half</button>
          <button class="btn" id="presetBtn1_3" onclick="setSimPreset('1:3')" title="1:3 Resolution (267x200)">1:3 (267×200)</button>
          <button class="btn" id="presetBtn1_4" onclick="setSimPreset('1:4')" title="1:4 Quarter Resolution (200x150)">1:4 Quarter</button>
          <button class="btn" id="presetBtn1_6" onclick="setSimPreset('1:6')" title="1:6 Sixth Resolution (133x100)">1:6 Sixth</button>
        </div>
      </div>

      <!-- Fixed Dimensions Presets (Grid Independent of Window Size) -->
      <div style="display:flex; flex-direction:column; gap:3px; margin-top:2px;">
        <div class="meta-row"><span>Fixed Grid Presets (Window-Independent)</span></div>
        <div class="btn-row" style="flex-wrap:wrap;">
          <button class="btn" id="presetBtn160" onclick="setSimPreset('160x120')" title="Fast 160x120 Grid">160×120</button>
          <button class="btn" id="presetBtn200" onclick="setSimPreset('200x150')" title="Compact 200x150 Grid">200×150</button>
          <button class="btn" id="presetBtn267" onclick="setSimPreset('267x200')" title="Classic 267x200 Grid">267×200</button>
          <button class="btn" id="presetBtn320" onclick="setSimPreset('320x240')" title="Smooth 320x240 Grid">320×240</button>
          <button class="btn" id="presetBtn400" onclick="setSimPreset('400x300')" title="Detailed 400x300 Grid">400×300</button>
          <button class="btn" id="presetBtn512" onclick="setSimPreset('512x384')" title="High-Density 512x384 Grid">512×384</button>
        </div>
      </div>

      <!-- Continuous Scale Slider -->
      <div style="display:flex; flex-direction:column; gap:3px; margin-top:2px;">
        <div class="meta-row"><span>Custom Scale Factor</span><span id="simScaleVal">1.00x (800×600)</span></div>
        <input type="range" id="simScaleSlider" min="0.15" max="1.50" step="0.01" value="1.00">
      </div>

      <!-- Zero-Artifact Filter Mode -->
      <div style="display:flex; flex-direction:column; gap:3px; margin-top:2px;">
        <div class="meta-row"><span>Zero-Artifact Resampling Filter</span></div>
        <div class="btn-row">
          <button class="btn active" id="filterBtnHermite" onclick="setSimFiltering(0)" title="Continuous C1 Hermite S-Curve (Zero blockiness, crisp organic CA features)">Smooth Hermite</button>
          <button class="btn" id="filterBtnBilinear" onclick="setSimFiltering(1)" title="Standard bilinear blend">Bilinear</button>
          <button class="btn" id="filterBtnNearest" onclick="setSimFiltering(2)" title="Clean pixel-art nearest neighbor grid">Nearest Grid</button>
        </div>
      </div>

      <!-- State Actions: Reset State -->
      <div class="btn-row" style="margin-top:6px; gap:6px;">
        <button class="toggle-btn active" id="resetSimulationStateBtn" style="flex:1.4; border-color:#ff9900; color:#ffaa33; font-weight:700;" onclick="resetSimulationStateBtn()" title="Clear canvas, backbuffer, and cleanly re-seed alive center patch in current resolution">🌀 Reset Simulation State</button>
      </div>

      <!-- NCA step scale -->
      <div style="display:flex; flex-direction:column; gap:3px; margin-top:2px;">
        <div class="meta-row"><span>NCA Step Scale</span><span id="ncaStepVal">1.0x</span></div>
        <input type="range" id="ncaStepSlider" min="0.2" max="2.0" step="0.05" value="1.0">
      </div>
    </div>

    <!-- Structure Imprint<!-- Structure Imprint (Giant Full-Screen Easter Egg) -->
    <div class="section">
      <div class="section-head">
        <span class="section-title">Giant Structure Imprint (Easter Egg)</span>
        <span id="imprintFeedback" class="val" style="color:var(--accent-cyan); font-size:0.75rem;">Ready</span>
      </div>
      <div style="display:flex; gap:8px;">
        <input type="text" id="imprintInput" maxlength="40" placeholder="Arbitrary message..." value="BYOB" style="flex:1;">
        <button class="toggle-btn active" onclick="triggerImprint()" style="white-space:nowrap;">🌟 Imprint Giant</button>
      </div>
      <div class="btn-row">
        <button class="btn" onclick="setImprintText('BYOB')">BYOB</button>
        <button class="btn" onclick="setImprintText('ANTIGRAVITY')">ANTIGRAVITY</button>
        <button class="btn" onclick="setImprintText('POLAR LIFE')">POLAR LIFE</button>
        <button class="btn" onclick="setImprintText('CELLULAR')">CELLULAR</button>
      </div>
      <div class="meta-row">
        <span>Rasterizer: Full-Screen Adaptive 8x8 Bitmap</span>
        <span>Auto Centered</span>
      </div>
    </div>

  </div>

  <script>
    let state = {
      scanning_speed: 0.40,
      decay_time: 0.9700,
      freq_min: 60,
      freq_max: 4000,
      scale: "log",
      scale_type: 0,
      auto_speed: true,
      rhythm_division: 2,
      detected_bpm: 120.0,
      auto_freq: true,
      auto_gain: 1.0,
      spectrogram_influence: 1.0,
      nca_topology: 1.0,
      topology: "polar",
      nca_pole_radial: -1.0,
      nca_pole_azimuthal: 1.0,
      reverse_pole: true,
      reverse_spin: false,
      nca_step_scale: 1.0,
      sim_width: 800,
      sim_height: 600,
      sim_scale_mode: 0,
      sim_scale: 1.0,
      sim_preset: "1:1",
      sim_filtering: 0,
      nca_perturb: 0.0,
      nca_seed: 0.0,
      nca_channel_view: -1,
      custom_channel_r: 0,
      custom_channel_g: 1,
      custom_channel_b: 2,
      pca_hue_shift: 0.0,
      spectrogram_scale: 1.0,
      auto_decay: true,
      decay_speed: 1.0
    };

    const connDot = document.getElementById('connDot');
    const speedSlider = document.getElementById('speedSlider');
    const specInfluenceSlider = document.getElementById('specInfluenceSlider');
    const specScaleSlider = document.getElementById('specScaleSlider');
    const decaySlider = document.getElementById('decaySlider');
    const freqMinSlider = document.getElementById('freqMinSlider');
    const freqMaxSlider = document.getElementById('freqMaxSlider');
    const topologySlider = document.getElementById('topologySlider');
    const ncaStepSlider = document.getElementById('ncaStepSlider');
    const simScaleSlider = document.getElementById('simScaleSlider');
    const imprintInput = document.getElementById('imprintInput');

    const speedVal = document.getElementById('speedVal');
    const specInfluenceVal = document.getElementById('specInfluenceVal');
    const specScaleVal = document.getElementById('specScaleVal');
    const decayVal = document.getElementById('decayVal');
    const decaySpeedVal = document.getElementById('decaySpeedVal');
    const autoDecayBtn = document.getElementById('autoDecayBtn');
    const autoDecayStatus = document.getElementById('autoDecayStatus');
    const freqRangeVal = document.getElementById('freqRangeVal');
    const minFreqVal = document.getElementById('minFreqVal');
    const maxFreqVal = document.getElementById('maxFreqVal');
    const scaleLabel = document.getElementById('scaleLabel');
    const bpmLabel = document.getElementById('bpmLabel');
    const gainLabel = document.getElementById('gainLabel');
    const topologyLabel = document.getElementById('topologyLabel');
    const topologyDesc = document.getElementById('topologyDesc');
    const ncaStepVal = document.getElementById('ncaStepVal');
    const simResBadge = document.getElementById('simResBadge');
    const simInfoLabel = document.getElementById('simInfoLabel');
    const simScaleVal = document.getElementById('simScaleVal');
    const imprintFeedback = document.getElementById('imprintFeedback');

    const autoSpeedBtn = document.getElementById('autoSpeedBtn');
    const autoFreqBtn = document.getElementById('autoFreqBtn');
    const reversePoleBtn = document.getElementById('reversePoleBtn');
    const reverseSpinBtn = document.getElementById('reverseSpinBtn');

    let isSending = false;
    let pendingUpdate = false;

    function updateUI(s) {
      if (s.scanning_speed !== undefined) {
        speedVal.innerText = Number(s.scanning_speed).toFixed(2) + ' rev/s';
        if (speedSlider && !speedSlider.matches(':active')) {
          speedSlider.value = s.scanning_speed;
        }
      }
      if (s.decay_time !== undefined) {
        state.decay_time = Number(s.decay_time);
        const dt = state.decay_time;
        decayVal.innerText = dt.toFixed(4);
        if (decaySlider && !decaySlider.matches(':active')) {
          decaySlider.value = dt;
        }
        document.getElementById('decayFastBtn')?.classList.toggle('active', Math.abs(dt - 0.9200) < 0.005);
        document.getElementById('decayMedBtn')?.classList.toggle('active', Math.abs(dt - 0.9700) < 0.005);
        document.getElementById('decayDefBtn')?.classList.toggle('active', Math.abs(dt - 0.9950) < 0.002);
        document.getElementById('decayLongBtn')?.classList.toggle('active', Math.abs(dt - 0.9990) < 0.002);
      }
      if (s.decay_speed !== undefined) {
        state.decay_speed = parseFloat(s.decay_speed);
        const ds = state.decay_speed;
        if (decaySpeedVal) decaySpeedVal.innerText = ds.toFixed(2) + 'x';
        document.getElementById('decaySpeedBtn05')?.classList.toggle('active', Math.abs(ds - 0.5) < 0.05);
        document.getElementById('decaySpeedBtn1')?.classList.toggle('active', Math.abs(ds - 1.0) < 0.05);
        document.getElementById('decaySpeedBtn2')?.classList.toggle('active', Math.abs(ds - 2.0) < 0.05);
        document.getElementById('decaySpeedBtn4')?.classList.toggle('active', Math.abs(ds - 4.0) < 0.05);
      }
      if (s.auto_decay !== undefined) {
        state.auto_decay = !!s.auto_decay;
        autoDecayBtn?.classList.toggle('active', state.auto_decay);
        if (autoDecayStatus) autoDecayStatus.innerText = state.auto_decay ? 'Calibrated to Speed' : 'Manual Decay';
      }
      if (s.freq_min !== undefined) minFreqVal.innerText = Math.round(s.freq_min) + ' Hz';
      if (s.freq_max !== undefined) maxFreqVal.innerText = Math.round(s.freq_max) + ' Hz';
      if (s.freq_min !== undefined && s.freq_max !== undefined) {
        freqRangeVal.innerText = Math.round(s.freq_min) + ' - ' + Math.round(s.freq_max) + ' Hz';
      }

      // Spectrogram influence & scanning line visibility
      if (s.spectrogram_influence !== undefined) {
        state.spectrogram_influence = s.spectrogram_influence;
        const inf = Number(s.spectrogram_influence);
        specInfluenceVal.innerText = inf.toFixed(2) + (inf <= 0.001 ? ' (Hidden / 0)' : '');
        if (specInfluenceSlider && !specInfluenceSlider.matches(':active')) {
          specInfluenceSlider.value = inf;
        }
        const desc = document.getElementById('specInfluenceDesc');
        if (desc) {
          desc.innerText = inf <= 0.001 ? 'Autonomous Evolution (Beam Hidden)' : ('Active Disturbance (' + Math.round(inf * 100) + '%)');
        }
        document.getElementById('specBtn0')?.classList.toggle('active', inf <= 0.001);
        document.getElementById('specBtn50')?.classList.toggle('active', Math.abs(inf - 0.5) < 0.05);
        document.getElementById('specBtn100')?.classList.toggle('active', Math.abs(inf - 1.0) < 0.05);
        document.getElementById('specBtn150')?.classList.toggle('active', Math.abs(inf - 1.5) < 0.05);
      }

      // Spectrogram scale (size: 0.5x, 1x, 2x, 4x) w/o disabling auto range
      if (s.spectrogram_scale !== undefined) {
        state.spectrogram_scale = parseFloat(s.spectrogram_scale);
        const sc = state.spectrogram_scale;
        if (specScaleVal) specScaleVal.innerText = sc.toFixed(2) + 'x';
        if (specScaleSlider && !specScaleSlider.matches(':active')) {
          specScaleSlider.value = sc;
        }
        document.getElementById('specScaleBtn05')?.classList.toggle('active', Math.abs(sc - 0.5) < 0.05);
        document.getElementById('specScaleBtn1')?.classList.toggle('active', Math.abs(sc - 1.0) < 0.05);
        document.getElementById('specScaleBtn2')?.classList.toggle('active', Math.abs(sc - 2.0) < 0.05);
        document.getElementById('specScaleBtn4')?.classList.toggle('active', Math.abs(sc - 4.0) < 0.05);
      }

      if (s.nca_step_scale !== undefined) {
        ncaStepVal.innerText = Number(s.nca_step_scale).toFixed(2) + 'x';
      }

      // Topology
      const topVal = parseFloat(s.nca_topology !== undefined ? s.nca_topology : 1.0);
      if (topologySlider && !topologySlider.matches(':active')) {
        topologySlider.value = topVal;
      }
      const revPole = (s.reverse_pole !== undefined) ? !!s.reverse_pole : (s.nca_pole_radial < 0);
      const revSpin = (s.reverse_spin !== undefined) ? !!s.reverse_spin : (s.nca_pole_azimuthal < 0);

      if (topVal <= 0.05) {
        topologyLabel.innerText = 'Cartesian (0.0)';
        topologyDesc.innerText = 'Cartesian ∇x, ∇y (Original byob)';
      } else if (topVal >= 0.95) {
        topologyLabel.innerText = revPole ? 'Polar (Inward Pole)' : 'Polar (Outward Pole)';
        topologyDesc.innerText = revPole ? 'Inward Sink ∇r, Azimuthal ∇θ' : 'Outward Source ∇r, Azimuthal ∇θ';
      } else {
        topologyLabel.innerText = 'Spiral (' + topVal.toFixed(2) + ')';
        topologyDesc.innerText = 'Rotated Sobel Field (' + Math.round(topVal * 90) + '°)';
      }

      document.getElementById('topoBtnCart')?.classList.toggle('active', topVal <= 0.1);
      document.getElementById('topoBtnSpiral')?.classList.toggle('active', topVal > 0.1 && topVal < 0.9);
      document.getElementById('topoBtnPolar')?.classList.toggle('active', topVal >= 0.9);

      if (reversePoleBtn) {
        reversePoleBtn.innerText = revPole ? '🔄 Reverse Pole: Inward' : '🔄 Reverse Pole: Outward';
        reversePoleBtn.classList.toggle('active', revPole);
      }

      if (reverseSpinBtn) {
        reverseSpinBtn.innerText = revSpin ? '🔃 Spin: CW (Rev)' : '🔃 Spin: CCW';
        reverseSpinBtn.classList.toggle('active', revSpin);
      }

      // Simulation Resolution UI
      const sw = s.sim_width || 800;
      const sh = s.sim_height || 600;
      const totalCells = (sw * sh) >= 1000 ? ((sw * sh) / 1000).toFixed(1) + 'k' : (sw * sh);
      const isRelative = (s.sim_scale_mode === 0);
      const presetName = s.sim_preset || (isRelative ? (sw === 800 && sh === 600 ? '1:1' : '1:3') : (sw + 'x' + sh));
      
      if (simResBadge) {
        simResBadge.innerText = `${sw} × ${sh} (${presetName})`;
      }
      if (simInfoLabel) {
        simInfoLabel.innerText = `${totalCells} cells • ${isRelative ? 'Relative ' + presetName : 'Fixed Grid'}`;
      }
      if (simScaleVal) {
        const sc = s.sim_scale !== undefined ? parseFloat(s.sim_scale) : 1.0;
        simScaleVal.innerText = `${sc.toFixed(2)}x (${sw}×${sh})`;
        if (simScaleSlider && !simScaleSlider.matches(':active')) {
          simScaleSlider.value = sc;
        }
      }

      // Highlight active relative preset buttons
      document.getElementById('presetBtn1_1')?.classList.toggle('active', presetName === '1:1');
      document.getElementById('presetBtn2_3')?.classList.toggle('active', presetName === '1:1.5' || presetName === '2:3');
      document.getElementById('presetBtn1_2')?.classList.toggle('active', presetName === '1:2' || presetName === 'half');
      document.getElementById('presetBtn1_3')?.classList.toggle('active', presetName === '1:3' || presetName === 'default');
      document.getElementById('presetBtn1_4')?.classList.toggle('active', presetName === '1:4' || presetName === 'quarter');
      document.getElementById('presetBtn1_6')?.classList.toggle('active', presetName === '1:6' || presetName === 'sixth');

      // Highlight active fixed preset buttons
      document.getElementById('presetBtn160')?.classList.toggle('active', presetName === '160x120');
      document.getElementById('presetBtn200')?.classList.toggle('active', presetName === '200x150');
      document.getElementById('presetBtn267')?.classList.toggle('active', presetName === '267x200');
      document.getElementById('presetBtn320')?.classList.toggle('active', presetName === '320x240');
      document.getElementById('presetBtn400')?.classList.toggle('active', presetName === '400x300');
      document.getElementById('presetBtn512')?.classList.toggle('active', presetName === '512x384');

      // Highlight active filtering buttons
      const filt = s.sim_filtering !== undefined ? parseInt(s.sim_filtering) : 0;
      document.getElementById('filterBtnHermite')?.classList.toggle('active', filt === 0);
      document.getElementById('filterBtnBilinear')?.classList.toggle('active', filt === 1);
      document.getElementById('filterBtnNearest')?.classList.toggle('active', filt === 2);

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
      document.getElementById('scaleBtnLog')?.classList.toggle('active', curScale === 'log');
      document.getElementById('scaleBtnMel')?.classList.toggle('active', curScale === 'mel');
      document.getElementById('scaleBtnLin')?.classList.toggle('active', curScale === 'linear');

      for (let i = 0; i <= 4; i++) {
        const btn = document.getElementById('divBtn' + i);
        if (btn) btn.classList.toggle('active', s.rhythm_division === i);
      }

      // Direct NCA Channel View & Custom Channel UI
      if (s.nca_channel_view !== undefined) {
        state.nca_channel_view = parseInt(s.nca_channel_view);
        const ch = state.nca_channel_view;
        const chLabel = document.getElementById('channelViewLabel');
        const chDesc = document.getElementById('channelViewDesc');
        if (chLabel) {
          if (ch === -1) {
            chLabel.innerText = 'Composite (CC)';
            if (chDesc) chDesc.innerText = 'Color corrected composite (Purple/Green)';
          } else if (ch === 12) {
            chLabel.innerText = '3 Channels (RGB)';
            if (chDesc) chDesc.innerText = 'Direct 3-channel RGB view (No Color Correction)';
          } else if (ch === 13) {
            chLabel.innerText = 'PCA RGB (3 PCs)';
            if (chDesc) chDesc.innerText = 'Top 3 Principal Components mapped to RGB (Rotatable Palette)';
          } else {
            chLabel.innerText = 'Channel ' + ch + ' (Grayscale)';
            if (chDesc) chDesc.innerText = 'Channel ' + ch + ' magnitude |state| (No Color Correction)';
          }
        }
        document.getElementById('chBtnComp')?.classList.toggle('active', ch === -1);
        document.getElementById('chBtnCustom')?.classList.toggle('active', ch === 12);
        document.getElementById('chBtnPca')?.classList.toggle('active', ch === 13);
        for (let c = 0; c < 12; c++) {
          document.getElementById('chBtn' + c)?.classList.toggle('active', ch === c);
        }
      }

      // Sync custom channels selectors
      if (s.custom_channel_r !== undefined) {
        state.custom_channel_r = parseInt(s.custom_channel_r);
        const el = document.getElementById('selChR');
        if (el && !el.matches(':active')) el.value = state.custom_channel_r;
      }
      if (s.custom_channel_g !== undefined) {
        state.custom_channel_g = parseInt(s.custom_channel_g);
        const el = document.getElementById('selChG');
        if (el && !el.matches(':active')) el.value = state.custom_channel_g;
      }
      if (s.custom_channel_b !== undefined) {
        state.custom_channel_b = parseInt(s.custom_channel_b);
        const el = document.getElementById('selChB');
        if (el && !el.matches(':active')) el.value = state.custom_channel_b;
      }
      const customDesc = document.getElementById('customChannelsDesc');
      if (customDesc) {
        const cr = state.custom_channel_r !== undefined ? state.custom_channel_r : 0;
        const cg = state.custom_channel_g !== undefined ? state.custom_channel_g : 1;
        const cb = state.custom_channel_b !== undefined ? state.custom_channel_b : 2;
        customDesc.innerText = `R: Ch ${cr} | G: Ch ${cg} | B: Ch ${cb}`;
      }

      // Sync PCA Hue
      if (s.pca_hue_shift !== undefined) {
        let deg = Math.round(parseFloat(s.pca_hue_shift)) % 360;
        if (deg < 0) deg += 360;
        state.pca_hue_shift = deg;
        const slider = document.getElementById('pcaHueSlider');
        if (slider && !slider.matches(':active')) slider.value = deg;
        const lbl = document.getElementById('pcaHueAngleLabel');
        if (lbl) lbl.innerText = deg + '°';
        const valTxt = document.getElementById('pcaHueVal');
        if (valTxt) valTxt.innerText = deg === 0 ? '0° (Original)' : deg + '°';
      }
    }

    function syncSliders() {
      speedSlider.value = state.scanning_speed;
      decaySlider.value = state.decay_time;
      if (specInfluenceSlider) specInfluenceSlider.value = state.spectrogram_influence !== undefined ? state.spectrogram_influence : 1.0;
      if (specScaleSlider) specScaleSlider.value = state.spectrogram_scale !== undefined ? state.spectrogram_scale : 1.0;
      freqMinSlider.value = state.freq_min;
      freqMaxSlider.value = state.freq_max;
      topologySlider.value = state.nca_topology;
      ncaStepSlider.value = state.nca_step_scale;
      if (simScaleSlider) simScaleSlider.value = state.sim_scale || 1.0;
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
          auto_decay: !!state.auto_decay,
          decay_speed: parseFloat(state.decay_speed !== undefined ? state.decay_speed : 1.0),
          spectrogram_influence: parseFloat(state.spectrogram_influence !== undefined ? state.spectrogram_influence : 1.0),
          spectrogram_scale: parseFloat(state.spectrogram_scale !== undefined ? state.spectrogram_scale : 1.0),
          freq_min: parseFloat(state.freq_min),
          freq_max: parseFloat(state.freq_max),
          scale: state.scale,
          scale_type: parseInt(state.scale_type),
          auto_speed: !!state.auto_speed,
          rhythm_division: parseInt(state.rhythm_division),
          auto_freq: !!state.auto_freq,
          nca_topology: parseFloat(state.nca_topology),
          nca_pole_radial: parseFloat(state.nca_pole_radial),
          nca_pole_azimuthal: parseFloat(state.nca_pole_azimuthal),
          reverse_pole: !!state.reverse_pole,
          reverse_spin: !!state.reverse_spin,
          nca_step_scale: parseFloat(state.nca_step_scale),
          nca_perturb: parseFloat(state.nca_perturb || 0.0),
          nca_seed: parseFloat(state.nca_seed || 0.0),
          nca_channel_view: parseInt(state.nca_channel_view !== undefined ? state.nca_channel_view : -1),
          custom_channel_r: parseInt(state.custom_channel_r !== undefined ? state.custom_channel_r : 0),
          custom_channel_g: parseInt(state.custom_channel_g !== undefined ? state.custom_channel_g : 1),
          custom_channel_b: parseInt(state.custom_channel_b !== undefined ? state.custom_channel_b : 2),
          pca_hue_shift: parseFloat(state.pca_hue_shift || 0.0)
        };
        const res = await fetch('/api/controls', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (res.ok) {
          const data = await res.json();
          connDot.className = 'conn-dot online';
          Object.assign(state, data);
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

    async function multiplySpeed(factor) {
      state.auto_speed = false;
      autoSpeedBtn.classList.remove('active');
      try {
        const res = await fetch('/api/controls', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ speed_multiplier: factor })
        });
        if (res.ok) {
          const data = await res.json();
          state.scanning_speed = data.scanning_speed;
          speedSlider.value = data.scanning_speed;
          updateUI(state);
        }
      } catch (e) {}
    }

    speedSlider.addEventListener('input', () => {
      state.scanning_speed = parseFloat(speedSlider.value);
      state.auto_speed = false;
      autoSpeedBtn.classList.remove('active');
      if (state.auto_decay) {
        const dspeed = parseFloat(state.decay_speed) || 1.0;
        state.decay_time = Math.max(0.50, Math.min(0.9995, Math.pow(0.9700, (state.scanning_speed / 0.50) * dspeed)));
      }
      updateUI(state);
      sendControls();
    });

    specInfluenceSlider.addEventListener('input', () => {
      state.spectrogram_influence = parseFloat(specInfluenceSlider.value);
      updateUI(state);
      sendControls();
    });

    if (specScaleSlider) {
      specScaleSlider.addEventListener('input', () => {
        state.spectrogram_scale = parseFloat(specScaleSlider.value);
        updateUI(state);
        sendControls();
      });
    }

    decaySlider.addEventListener('input', () => {
      state.decay_time = parseFloat(decaySlider.value);
      state.auto_decay = false;
      autoDecayBtn?.classList.remove('active');
      if (autoDecayStatus) autoDecayStatus.innerText = 'Manual Decay';
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

    topologySlider.addEventListener('input', () => {
      setTopology(parseFloat(topologySlider.value));
    });

    ncaStepSlider.addEventListener('input', () => {
      state.nca_step_scale = parseFloat(ncaStepSlider.value);
      updateUI(state);
      sendControls();
    });

    if (simScaleSlider) {
      simScaleSlider.addEventListener('input', () => {
        const sc = parseFloat(simScaleSlider.value);
        state.sim_scale = sc;
        const sw = Math.round(800 * sc);
        const sh = Math.round(600 * sc);
        if (simScaleVal) simScaleVal.innerText = `${sc.toFixed(2)}x (${sw}×${sh})`;
      });
      simScaleSlider.addEventListener('change', () => {
        setSimScale(simScaleSlider.value);
      });
    }

    function setTopology(v) {
      state.nca_topology = v;
      topologySlider.value = v;
      updateUI(state);
      sendControls();
    }

    function toggleReversePole() {
      state.reverse_pole = !state.reverse_pole;
      state.nca_pole_radial = state.reverse_pole ? -1.0 : 1.0;
      updateUI(state);
      sendControls();
    }

    function toggleReverseSpin() {
      state.reverse_spin = !state.reverse_spin;
      state.nca_pole_azimuthal = state.reverse_spin ? -1.0 : 1.0;
      updateUI(state);
      sendControls();
    }

    function setDecay(v) {
      decaySlider.value = v;
      state.decay_time = v;
      state.auto_decay = false;
      autoDecayBtn?.classList.remove('active');
      if (autoDecayStatus) autoDecayStatus.innerText = 'Manual Decay';
      updateUI(state);
      sendControls();
    }

    function toggleAutoDecay() {
      state.auto_decay = !state.auto_decay;
      autoDecayBtn?.classList.toggle('active', state.auto_decay);
      if (state.auto_decay) {
        const speed = Math.max(0.05, parseFloat(state.scanning_speed) || 0.50);
        const dspeed = parseFloat(state.decay_speed) || 1.0;
        state.decay_time = Math.max(0.50, Math.min(0.9995, Math.pow(0.9700, (speed / 0.50) * dspeed)));
      }
      if (autoDecayStatus) autoDecayStatus.innerText = state.auto_decay ? 'Calibrated to Speed' : 'Manual Decay';
      updateUI(state);
      sendControls();
    }

    function setDecaySpeed(mult) {
      state.decay_speed = parseFloat(mult);
      if (state.auto_decay) {
        const speed = Math.max(0.05, parseFloat(state.scanning_speed) || 0.50);
        state.decay_time = Math.max(0.50, Math.min(0.9995, Math.pow(0.9700, (speed / 0.50) * state.decay_speed)));
      } else {
        state.decay_time = Math.max(0.50, Math.min(0.9995, Math.pow(0.9700, state.decay_speed)));
      }
      updateUI(state);
      sendControls();
    }

    function multiplyDecaySpeed(factor) {
      const cur = parseFloat(state.decay_speed) || 1.0;
      setDecaySpeed(cur * factor);
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

    // --- Direct NCA Channel View & Custom Channel Actions ---
    async function setNcaChannel(ch) {
      state.nca_channel_view = parseInt(ch);
      const payload = { channel: state.nca_channel_view };
      if (state.nca_channel_view === 12) {
        payload.custom_r = parseInt(document.getElementById('selChR')?.value || 0);
        payload.custom_g = parseInt(document.getElementById('selChG')?.value || 1);
        payload.custom_b = parseInt(document.getElementById('selChB')?.value || 2);
      }
      try {
        const res = await fetch('/api/channel', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        if (res.ok) {
          const data = await res.json();
          Object.assign(state, data);
          updateUI(state);
        }
      } catch (e) {
        sendControls();
      }
    }

    async function updateCustomChannels() {
      const r = parseInt(document.getElementById('selChR').value);
      const g = parseInt(document.getElementById('selChG').value);
      const b = parseInt(document.getElementById('selChB').value);
      state.custom_channel_r = r;
      state.custom_channel_g = g;
      state.custom_channel_b = b;
      state.nca_channel_view = 12;
      try {
        const res = await fetch('/api/channel', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            channel: 12,
            custom_r: r,
            custom_g: g,
            custom_b: b
          })
        });
        if (res.ok) {
          const data = await res.json();
          Object.assign(state, data);
          updateUI(state);
        }
      } catch (e) {
        sendControls();
      }
    }

    // --- PCA Palette Actions ---
    async function rerollPcaPalette() {
      try {
        const res = await fetch('/api/pca/palette', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ reroll: true })
        });
        if (res.ok) {
          const data = await res.json();
          Object.assign(state, data);
          updateUI(state);
        }
      } catch (e) {}
    }

    async function shiftPcaPalette(deltaDeg) {
      try {
        const res = await fetch('/api/pca/palette', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ shift: deltaDeg })
        });
        if (res.ok) {
          const data = await res.json();
          Object.assign(state, data);
          updateUI(state);
        }
      } catch (e) {}
    }

    async function onPcaHueSlider(valDeg) {
      const deg = parseFloat(valDeg);
      state.pca_hue_shift = deg;
      const degLabel = document.getElementById('pcaHueAngleLabel');
      if (degLabel) degLabel.innerText = Math.round(deg) + '°';
      const valText = document.getElementById('pcaHueVal');
      if (valText) valText.innerText = Math.round(deg) + '°';
      try {
        const res = await fetch('/api/pca/palette', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ hue: deg })
        });
        if (res.ok) {
          const data = await res.json();
          Object.assign(state, data);
        }
      } catch (e) {}
    }

    // --- Simulation Resolution Actions ---
    async function setSimPreset(preset) {
      try {
        const res = await fetch('/api/resolution', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ preset: preset, resample: true })
        });
        if (res.ok) {
          const data = await res.json();
          Object.assign(state, data);
          updateUI(state);
        }
      } catch (e) {}
    }

    async function setSimScale(scale) {
      try {
        const res = await fetch('/api/resolution', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ scale: parseFloat(scale), resample: true })
        });
        if (res.ok) {
          const data = await res.json();
          Object.assign(state, data);
          updateUI(state);
        }
      } catch (e) {}
    }

    async function setSimFiltering(mode) {
      try {
        const res = await fetch('/api/resolution', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ filtering: parseInt(mode) })
        });
        if (res.ok) {
          const data = await res.json();
          Object.assign(state, data);
          updateUI(state);
        }
      } catch (e) {}
    }

    async function resetSimulationStateBtn() {
      try {
        const label = document.getElementById('dimStatusLabel');
        if (label) label.innerText = 'Resetting State...';
        await fetch('/api/reset_state', { method: 'POST' });
        setTimeout(() => {
          if (label) label.innerText = 'Active (100% Energy)';
        }, 1200);
      } catch (e) {}
    }

    function setSpectrogramInfluence(v) {
      state.spectrogram_influence = v;
      if (specInfluenceSlider) specInfluenceSlider.value = v;
      updateUI(state);
      sendControls();
    }

    function setSpectrogramScale(v) {
      state.spectrogram_scale = parseFloat(v);
      if (specScaleSlider) specScaleSlider.value = v;
      updateUI(state);
      sendControls();
    }

    // --- Giant Easter Egg Structure Imprint ---
    function setImprintText(txt) {
      imprintInput.value = txt;
    }

    async function triggerImprint() {
      const msg = imprintInput.value.trim() || "BYOB";
      imprintFeedback.innerText = "Imprinting...";
      try {
        const res = await fetch('/api/imprint', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: msg })
        });
        if (res.ok) {
          imprintFeedback.innerText = "Imprinted: " + msg;
          setTimeout(() => { imprintFeedback.innerText = "Ready"; }, 3500);
        }
      } catch (e) {
        imprintFeedback.innerText = "Error";
      }
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
            if (data.auto_decay) {
              state.decay_time = data.decay_time;
              decaySlider.value = data.decay_time;
            }
            Object.assign(state, data);
            updateUI(state);
          }
        }
      } catch (err) {
        connDot.className = 'conn-dot';
      }
    }

    syncSliders();
    pollStatus();
    setInterval(pollStatus, 800);
  </script>
</body>
</html>
)rawliteral";
}

#endif // RADAR_WEB_SERVER_H
