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

    // Waveform visualization options:
    // 0 = Off / Spectrogram only
    // 1 = Auto (switch to waveform when rhythm is not established)
    // 2 = Always Waveform
    std::atomic<int> waveformMode{1};
    std::atomic<bool> rhythmEstablished{false};
    mutable std::atomic<bool> waveformActive{true};

    // NCA weight perturbation controls
    std::atomic<float> ncaPerturb{0.0f};      // 0.0 (unperturbed) to 1.0 (heavy perturbation)
    std::atomic<float> ncaSeed{0.0f};         // Mutation seed
    std::atomic<float> ncaStepScale{1.0f};    // NCA reaction step scale (default 1.0)

    // Topology control: 0.0 = Cartesian, 1.0 = Polar (pole in center)
    std::atomic<float> ncaTopology{0.0f};

    // Human-Assisted Genetic Algorithm state
    std::atomic<int> ncaGeneration{0};
    std::atomic<bool> ncaHasCandidate{false};
    std::atomic<int> ncaMutatedCount{0};
    std::atomic<float> ncaMutationScale{0.15f};
    std::atomic<float> ncaMutationRate{0.25f};

    // Action triggers for Genetic Algorithm
    std::atomic<bool> geneticActionPending{false};
    std::mutex geneticMutex;
    std::string pendingGeneticAction;
    std::string pendingGeneticTarget{"all"};
    float pendingGeneticScale{0.15f};
    float pendingGeneticRate{0.25f};

    // Action triggers for Structure Imprint Easter Egg
    std::atomic<bool> hasPendingImprint{false};
    std::mutex imprintMutex;
    std::string pendingImprintMessage;

    void requestImprint(const std::string& msg) {
        std::lock_guard<std::mutex> lock(imprintMutex);
        pendingImprintMessage = msg;
        hasPendingImprint.store(true);
    }

    void requestGenetic(const std::string& action, const std::string& target, float scale, float rate) {
        std::lock_guard<std::mutex> lock(geneticMutex);
        pendingGeneticAction = action;
        pendingGeneticTarget = target;
        pendingGeneticScale = scale;
        pendingGeneticRate = rate;
        geneticActionPending.store(true);
    }

    struct GeneticRequest {
        std::string action;
        std::string target{"all"};
        float scale{0.15f};
        float rate{0.25f};
    };

    bool popPendingImprint(std::string& outMsg) {
        if (!hasPendingImprint.load()) return false;
        std::lock_guard<std::mutex> lock(imprintMutex);
        if (!hasPendingImprint.load()) return false;
        outMsg = pendingImprintMessage;
        hasPendingImprint.store(false);
        return true;
    }

    bool popPendingGenetic(GeneticRequest& req) {
        if (!geneticActionPending.load()) return false;
        std::lock_guard<std::mutex> lock(geneticMutex);
        if (!geneticActionPending.load()) return false;
        req.action = pendingGeneticAction;
        req.target = pendingGeneticTarget;
        req.scale = pendingGeneticScale;
        req.rate = pendingGeneticRate;
        geneticActionPending.store(false);
        return true;
    }

    std::string scaleName() const {
        int s = scaleType.load();
        if (s == 1) return "mel";
        if (s == 2) return "linear";
        return "log";
    }

    std::string waveformModeName() const {
        int m = waveformMode.load();
        if (m == 0) return "off";
        if (m == 2) return "always";
        return "auto";
    }

    bool isWaveformActive() const {
        int m = waveformMode.load();
        if (m == 2) return true;
        if (m == 0) return false;
        return !rhythmEstablished.load();
    }

    std::string toJson() const {
        bool active = isWaveformActive();
        waveformActive.store(active);
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
        ss << "  \"waveform_mode\": " << waveformMode.load() << ",\n";
        ss << "  \"waveform_mode_name\": \"" << waveformModeName() << "\",\n";
        ss << "  \"rhythm_established\": " << (rhythmEstablished.load() ? "true" : "false") << ",\n";
        ss << "  \"waveform_active\": " << (active ? "true" : "false") << ",\n";
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
        ss << "  \"genetic_generation\": " << ncaGeneration.load() << ",\n";
        ss << "  \"genetic_has_candidate\": " << (ncaHasCandidate.load() ? "true" : "false") << ",\n";
        ss << "  \"genetic_mutated_count\": " << ncaMutatedCount.load() << ",\n";
        ss << "  \"genetic_mutation_scale\": " << ncaMutationScale.load() << ",\n";
        ss << "  \"genetic_mutation_rate\": " << ncaMutationRate.load() << ",\n";
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

        // Human-Assisted Genetic Algorithm endpoint
        if (cleanPath == "/api/genetic") {
            if (method == "POST") {
                size_t bodyPos = request.find("\r\n\r\n");
                std::string body = (bodyPos != std::string::npos) ? request.substr(bodyPos + 4) : "";
                std::string action = "mutate";
                std::string target = "all";
                float scale = 0.15f;
                float rate = 0.25f;
                extractJsonString(body, "action", action);
                extractJsonString(body, "target", target);
                extractJsonFloat(body, "scale", scale);
                extractJsonFloat(body, "rate", rate);

                m_config.requestGenetic(action, target, scale, rate);
                std::this_thread::sleep_for(std::chrono::milliseconds(25));
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

            // Speed multiplier (e.g. 0.25x, 0.5x, 2.0x, 4.0x)
            if (extractJsonFloat(body, "speed_multiplier", val)) {
                if (val > 0.001f && val <= 10.0f) {
                    float cur = m_config.scanningSpeed.load();
                    float newSpeed = std::max(0.01f, std::min(10.0f, cur * val));
                    m_config.scanningSpeed.store(newSpeed);
                    m_config.autoSpeed.store(false);
                }
            }

            // Waveform mode setting (0: off, 1: auto, 2: always)
            if (extractJsonString(body, "waveform_mode", strVal)) {
                if (strVal == "auto") m_config.waveformMode.store(1);
                else if (strVal == "always" || strVal == "on" || strVal == "waveform") m_config.waveformMode.store(2);
                else if (strVal == "off" || strVal == "spectrogram") m_config.waveformMode.store(0);
                m_config.waveformActive.store(m_config.isWaveformActive());
            } else if (extractJsonInt(body, "waveform_mode", intVal)) {
                if (intVal >= 0 && intVal <= 2) {
                    m_config.waveformMode.store(intVal);
                    m_config.waveformActive.store(m_config.isWaveformActive());
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
            if (extractJsonString(body, "genetic_action", strVal)) {
                std::string target = "all";
                float scale = 0.15f;
                float rate = 0.25f;
                extractJsonString(body, "genetic_target", target);
                extractJsonFloat(body, "genetic_scale", scale);
                extractJsonFloat(body, "genetic_rate", rate);
                m_config.requestGenetic(strVal, target, scale, rate);
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
            }
            if (extractJsonString(body, "imprint_message", strVal)) {
                m_config.requestImprint(strVal);
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

    <!-- NCA Topology: Cartesian vs Polar (Rotated Sobel Filters) -->
    <div class="section">
      <div class="section-head">
        <span class="section-title">NCA Topology</span>
        <span id="topologyLabel" class="val">Cartesian (0.0)</span>
      </div>
      <div class="btn-row">
        <button class="btn active" id="topoBtnCart" onclick="setTopology(0.0)">Cartesian</button>
        <button class="btn" id="topoBtnSpiral" onclick="setTopology(0.5)">Spiral / Vortex</button>
        <button class="btn" id="topoBtnPolar" onclick="setTopology(1.0)">Polar (Pole Center)</button>
      </div>
      <input type="range" id="topologySlider" min="0.0" max="1.0" step="0.02" value="0.0">
      <div class="meta-row">
        <span>Sobel Filter Field:</span>
        <span id="topologyDesc">Cartesian ∇x, ∇y</span>
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

      <div style="display:flex; flex-direction:column; gap:3px;">
        <div class="meta-row"><span>Min Freq</span><span id="minFreqVal">100 Hz</span></div>
        <input type="range" id="freqMinSlider" min="20" max="2000" step="10" value="100">
      </div>

      <div style="display:flex; flex-direction:column; gap:3px; margin-top:2px;">
        <div class="meta-row"><span>Max Freq</span><span id="maxFreqVal">10000 Hz</span></div>
        <input type="range" id="freqMaxSlider" min="200" max="20000" step="100" value="10000">
      </div>

      <div style="display:flex; justify-content:space-between; align-items:center; gap:8px; margin-top:2px;">
        <button id="autoFreqBtn" class="toggle-btn" onclick="toggleAutoFreq()">
          ⚡ Auto Freq & Scaling
        </button>
        <span id="gainLabel" class="meta-row" style="margin:0;">Gain: 1.0x</span>
      </div>
    </div>

    <!-- Human-Assisted Genetic Evolution -->
    <div class="section">
      <div class="section-head">
        <span class="section-title">Genetic Algorithm</span>
        <span id="geneticGenLabel" class="val">Gen: 0</span>
      </div>

      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span id="candidateBadge" class="badge badge-stable">Baseline Stable</span>
        <span id="mutatedCountLabel" class="meta-row" style="margin:0;">Mutated: 0 weights</span>
      </div>

      <!-- Target subsystem selector -->
      <div style="display:flex; flex-direction:column; gap:3px;">
        <div class="meta-row"><span>Target Subsystem</span><span id="targetNameLabel">All (588)</span></div>
        <div class="btn-row">
          <button class="btn active" id="targetBtnAll" onclick="setGeneticTarget('all')">All</button>
          <button class="btn" id="targetBtnPercep" onclick="setGeneticTarget('perception')">Perception</button>
          <button class="btn" id="targetBtnState" onclick="setGeneticTarget('state')">State</button>
          <button class="btn" id="targetBtnNonlin" onclick="setGeneticTarget('nonlinear')">Nonlinear</button>
          <button class="btn" id="targetBtnBias" onclick="setGeneticTarget('biases')">Biases</button>
        </div>
      </div>

      <!-- Mutation Scale & Rate sliders -->
      <div style="display:flex; flex-direction:column; gap:3px; margin-top:2px;">
        <div class="meta-row"><span>Mutation Scale</span><span id="mutScaleVal">0.15</span></div>
        <input type="range" id="mutScaleSlider" min="0.02" max="0.50" step="0.01" value="0.15">
      </div>
      <div style="display:flex; flex-direction:column; gap:3px; margin-top:2px;">
        <div class="meta-row"><span>Mutation Rate</span><span id="mutRateVal">25%</span></div>
        <input type="range" id="mutRateSlider" min="0.05" max="0.60" step="0.05" value="0.25">
      </div>

      <!-- Genetic Actions -->
      <div class="btn-row" style="margin-top:4px;">
        <button class="toggle-btn" style="flex:2;" onclick="mutateCandidate()">🎲 Mutate Candidate</button>
        <button class="btn" id="cancelBtn" style="flex:1; border-color:#ff4466; color:#ff7788;" onclick="cancelCandidate()" title="Cancel candidate and revert to baseline">❌ Cancel</button>
        <button class="btn" id="reinforceBtn" style="flex:1.2; border-color:var(--accent); color:var(--accent); font-weight:700;" onclick="reinforceCandidate()" title="Lock-in candidate as next generation">✨ Reinforce</button>
      </div>

      <div class="btn-row">
        <button class="btn" onclick="rollbackGeneration()" title="Rollback to previous generation">⏮ Rollback</button>
        <button class="btn" onclick="resetToGen0()" title="Reset to original starting NCA weights">↺ Reset Gen 0</button>
      </div>

      <!-- NCA step scale -->
      <div style="display:flex; flex-direction:column; gap:3px; margin-top:2px;">
        <div class="meta-row"><span>NCA Step Scale</span><span id="ncaStepVal">1.0x</span></div>
        <input type="range" id="ncaStepSlider" min="0.2" max="2.0" step="0.05" value="1.0">
      </div>
    </div>

    <!-- Structure Imprint (Easter Egg) -->
    <div class="section">
      <div class="section-head">
        <span class="section-title">Structure Imprint (Easter Egg)</span>
        <span id="imprintFeedback" class="val" style="color:var(--accent-cyan); font-size:0.75rem;">Ready</span>
      </div>
      <div style="display:flex; gap:8px;">
        <input type="text" id="imprintInput" maxlength="32" placeholder="Arbitrary message..." value="BYOB" style="flex:1;">
        <button class="toggle-btn active" onclick="triggerImprint()" style="white-space:nowrap;">🌟 Imprint</button>
      </div>
      <div class="btn-row">
        <button class="btn" onclick="setImprintText('BYOB')">BYOB</button>
        <button class="btn" onclick="setImprintText('ANTIGRAVITY')">ANTIGRAVITY</button>
        <button class="btn" onclick="setImprintText('POLAR LIFE')">POLAR LIFE</button>
        <button class="btn" onclick="setImprintText('CELLULAR')">CELLULAR</button>
      </div>
    </div>

  </div>

  <script>
    let state = {
      scanning_speed: 0.40,
      decay_time: 0.9950,
      freq_min: 100,
      freq_max: 10000,
      scale: "log",
      scale_type: 0,
      auto_speed: false,
      rhythm_division: 2,
      detected_bpm: 120.0,
      auto_freq: false,
      auto_gain: 1.0,
      nca_topology: 0.0,
      topology: "cartesian",
      nca_step_scale: 1.0,
      genetic_generation: 0,
      genetic_has_candidate: false,
      genetic_mutated_count: 0,
      genetic_mutation_scale: 0.15,
      genetic_mutation_rate: 0.25,
      // Retain for compatibility
      nca_perturb: 0.0,
      nca_seed: 0.0,
      waveform_mode: 0
    };

    let geneticTarget = "all";

    const connDot = document.getElementById('connDot');
    const speedSlider = document.getElementById('speedSlider');
    const decaySlider = document.getElementById('decaySlider');
    const freqMinSlider = document.getElementById('freqMinSlider');
    const freqMaxSlider = document.getElementById('freqMaxSlider');
    const topologySlider = document.getElementById('topologySlider');
    const ncaStepSlider = document.getElementById('ncaStepSlider');
    const mutScaleSlider = document.getElementById('mutScaleSlider');
    const mutRateSlider = document.getElementById('mutRateSlider');
    const imprintInput = document.getElementById('imprintInput');

    const speedVal = document.getElementById('speedVal');
    const decayVal = document.getElementById('decayVal');
    const freqRangeVal = document.getElementById('freqRangeVal');
    const minFreqVal = document.getElementById('minFreqVal');
    const maxFreqVal = document.getElementById('maxFreqVal');
    const scaleLabel = document.getElementById('scaleLabel');
    const bpmLabel = document.getElementById('bpmLabel');
    const gainLabel = document.getElementById('gainLabel');
    const topologyLabel = document.getElementById('topologyLabel');
    const topologyDesc = document.getElementById('topologyDesc');
    const ncaStepVal = document.getElementById('ncaStepVal');
    const geneticGenLabel = document.getElementById('geneticGenLabel');
    const candidateBadge = document.getElementById('candidateBadge');
    const mutatedCountLabel = document.getElementById('mutatedCountLabel');
    const targetNameLabel = document.getElementById('targetNameLabel');
    const mutScaleVal = document.getElementById('mutScaleVal');
    const mutRateVal = document.getElementById('mutRateVal');
    const imprintFeedback = document.getElementById('imprintFeedback');

    const autoSpeedBtn = document.getElementById('autoSpeedBtn');
    const autoFreqBtn = document.getElementById('autoFreqBtn');
    const cancelBtn = document.getElementById('cancelBtn');
    const reinforceBtn = document.getElementById('reinforceBtn');

    let isSending = false;
    let pendingUpdate = false;

    function updateUI(s) {
      speedVal.innerText = Number(s.scanning_speed).toFixed(2) + ' rev/s';
      decayVal.innerText = Number(s.decay_time).toFixed(4);
      minFreqVal.innerText = Math.round(s.freq_min) + ' Hz';
      maxFreqVal.innerText = Math.round(s.freq_max) + ' Hz';
      freqRangeVal.innerText = Math.round(s.freq_min) + ' - ' + Math.round(s.freq_max) + ' Hz';

      ncaStepVal.innerText = Number(s.nca_step_scale || 1.0).toFixed(2) + 'x';

      // Topology
      const topVal = parseFloat(s.nca_topology || 0.0);
      topologySlider.value = topVal;
      if (topVal <= 0.05) {
        topologyLabel.innerText = 'Cartesian (0.0)';
        topologyDesc.innerText = 'Cartesian ∇x, ∇y';
      } else if (topVal >= 0.95) {
        topologyLabel.innerText = 'Polar (1.0)';
        topologyDesc.innerText = 'Radial ∇r, Azimuthal ∇θ';
      } else {
        topologyLabel.innerText = 'Spiral (' + topVal.toFixed(2) + ')';
        topologyDesc.innerText = 'Rotated Sobel Field (' + Math.round(topVal * 90) + '°)';
      }
      document.getElementById('topoBtnCart').classList.toggle('active', topVal <= 0.1);
      document.getElementById('topoBtnSpiral').classList.toggle('active', topVal > 0.1 && topVal < 0.9);
      document.getElementById('topoBtnPolar').classList.toggle('active', topVal >= 0.9);

      // Genetic Algorithm state
      geneticGenLabel.innerText = 'Gen: ' + (s.genetic_generation !== undefined ? s.genetic_generation : 0);
      const hasCand = !!s.genetic_has_candidate;
      if (hasCand) {
        candidateBadge.className = 'badge badge-candidate';
        candidateBadge.innerText = 'Candidate Live';
        cancelBtn.style.opacity = '1.0';
        reinforceBtn.style.opacity = '1.0';
      } else {
        candidateBadge.className = 'badge badge-stable';
        candidateBadge.innerText = 'Baseline Stable';
        cancelBtn.style.opacity = '0.4';
        reinforceBtn.style.opacity = '0.4';
      }
      mutatedCountLabel.innerText = 'Mutated: ' + (s.genetic_mutated_count || 0) + ' weights';

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
      topologySlider.value = state.nca_topology;
      ncaStepSlider.value = state.nca_step_scale;
      mutScaleSlider.value = state.genetic_mutation_scale;
      mutRateSlider.value = state.genetic_mutation_rate;
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
          auto_freq: !!state.auto_freq,
          nca_topology: parseFloat(state.nca_topology),
          nca_step_scale: parseFloat(state.nca_step_scale),
          nca_perturb: parseFloat(state.nca_perturb || 0.0),
          nca_seed: parseFloat(state.nca_seed || 0.0)
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

    topologySlider.addEventListener('input', () => {
      setTopology(parseFloat(topologySlider.value));
    });

    ncaStepSlider.addEventListener('input', () => {
      state.nca_step_scale = parseFloat(ncaStepSlider.value);
      updateUI(state);
      sendControls();
    });

    mutScaleSlider.addEventListener('input', () => {
      state.genetic_mutation_scale = parseFloat(mutScaleSlider.value);
      mutScaleVal.innerText = state.genetic_mutation_scale.toFixed(2);
    });

    mutRateSlider.addEventListener('input', () => {
      state.genetic_mutation_rate = parseFloat(mutRateSlider.value);
      mutRateVal.innerText = Math.round(state.genetic_mutation_rate * 100) + '%';
    });

    function setTopology(v) {
      state.nca_topology = v;
      topologySlider.value = v;
      updateUI(state);
      sendControls();
    }

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

    // --- Genetic Algorithm Actions ---
    function setGeneticTarget(tgt) {
      geneticTarget = tgt;
      const names = {
        all: "All (588)",
        perception: "Perception (Sobel/Lap)",
        state: "State Recurrence",
        nonlinear: "Nonlinear (|xs|, |ps|)",
        biases: "Biases (12)"
      };
      targetNameLabel.innerText = names[tgt] || tgt;
      document.getElementById('targetBtnAll').classList.toggle('active', tgt === 'all');
      document.getElementById('targetBtnPercep').classList.toggle('active', tgt === 'perception');
      document.getElementById('targetBtnState').classList.toggle('active', tgt === 'state');
      document.getElementById('targetBtnNonlin').classList.toggle('active', tgt === 'nonlinear');
      document.getElementById('targetBtnBias').classList.toggle('active', tgt === 'biases');
    }

    async function mutateCandidate() {
      try {
        const res = await fetch('/api/genetic', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            action: 'mutate',
            target: geneticTarget,
            scale: parseFloat(state.genetic_mutation_scale),
            rate: parseFloat(state.genetic_mutation_rate)
          })
        });
        if (res.ok) {
          const data = await res.json();
          Object.assign(state, data);
          updateUI(state);
        }
      } catch (e) {}
    }

    async function cancelCandidate() {
      try {
        const res = await fetch('/api/genetic', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'cancel' })
        });
        if (res.ok) {
          const data = await res.json();
          Object.assign(state, data);
          updateUI(state);
        }
      } catch (e) {}
    }

    async function reinforceCandidate() {
      try {
        const res = await fetch('/api/genetic', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'reinforce' })
        });
        if (res.ok) {
          const data = await res.json();
          Object.assign(state, data);
          updateUI(state);
        }
      } catch (e) {}
    }

    async function rollbackGeneration() {
      try {
        const res = await fetch('/api/genetic', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'rollback' })
        });
        if (res.ok) {
          const data = await res.json();
          Object.assign(state, data);
          updateUI(state);
        }
      } catch (e) {}
    }

    async function resetToGen0() {
      try {
        const res = await fetch('/api/genetic', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ action: 'reset' })
        });
        if (res.ok) {
          const data = await res.json();
          Object.assign(state, data);
          updateUI(state);
        }
      } catch (e) {}
    }

    // --- Easter Egg Structure Imprint ---
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
          setTimeout(() => { imprintFeedback.innerText = "Ready"; }, 3000);
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
    setInterval(pollStatus, 1200);
  </script>
</body>
</html>
)rawliteral";
}

#endif // RADAR_WEB_SERVER_H
