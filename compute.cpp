#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <pulse/pulseaudio.h>
#include <pulse/simple.h>
#include <pulse/error.h>
#include <thread>
#include <chrono>
#include <mutex>
#include <vector>
#include <complex>
#include <fftw3.h>
#include <cmath>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>

#include "shader_c.h"
#include "shader_m.h"
#include "web_server.h"

// Global radar configuration and web server
RadarConfig g_radarConfig;
RadarWebServer* g_webServer = nullptr;
int WEB_PORT = 8080;
bool ENABLE_WEB = true;

// Global texture and buffer handles for recreation
unsigned int outputTexture, audioTexture, cameraTexture, backbufferTexture;
unsigned int stateBuffer, positionBuffer, velocityBuffer, trailGridBuffer, agentGridBuffer, angleBuffer;
unsigned int uniformBuffer;
bool texturesInitialized = false;

unsigned int frameCounter;
float currentFrame;
void recreateTextures(int width, int height);

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
    extern unsigned int SCR_WIDTH, SCR_HEIGHT, TEXTURE_WIDTH, TEXTURE_HEIGHT;
    SCR_WIDTH = width;
    SCR_HEIGHT = height;
    TEXTURE_WIDTH = width;
    TEXTURE_HEIGHT = height;
    
    std::cout << "Window resized to: " << width << "x" << height << std::endl;
    
    if (texturesInitialized) {
        recreateTextures(width, height);
    }
}

void recreateTextures(int width, int height) {
    // Recreate main texture
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, outputTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
    glBindImageTexture(0, outputTexture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
    
    // Recreate camera texture
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, cameraTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGR, GL_UNSIGNED_BYTE, NULL);
    glBindImageTexture(2, cameraTexture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8);
    
    // Recreate backbuffer texture
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, backbufferTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, NULL);
    glBindImageTexture(3, backbufferTexture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);
    
    // Recreate simulation state buffer
    const unsigned int N_CHANNELS = 12;
    const size_t stateBufferSize = width * height * N_CHANNELS * sizeof(float);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, stateBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, stateBufferSize, NULL, GL_DYNAMIC_DRAW);
    
    // Recreate trail grid buffer (uses width as grid size)
    const unsigned int GRID_SIZE = width;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, trailGridBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, GRID_SIZE * GRID_SIZE * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    
    // Recreate agent grid buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, agentGridBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, GRID_SIZE * GRID_SIZE * 4 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    
    // Clear agent position buffer to force reinitialization at new resolution
    const unsigned int AGENT_COUNT = 10000;
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, positionBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, AGENT_COUNT * 2 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    
    // Clear agent velocity buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, velocityBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, AGENT_COUNT * 2 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    
    // Clear angle buffer
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, angleBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, AGENT_COUNT * sizeof(float), NULL, GL_DYNAMIC_DRAW);

    // Reset uniforms
    currentFrame = 0.;
    frameCounter = 0;

    
    std::cout << "Textures and buffers recreated for resolution: " << width << "x" << height << std::endl;
}

void renderQuad();

// Screen resolution will be set dynamically
unsigned int SCR_WIDTH;
unsigned int SCR_HEIGHT;
unsigned int TEXTURE_WIDTH, TEXTURE_HEIGHT;

int CAMERA_DEV = -1;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;
const float FPS_CAP = 60.0f;
const float FRAME_TIME = 1.0f / FPS_CAP;

// Audio constants - Fixed buffer size mismatch
const int SAMPLE_RATE = 44100;
const int BUFFER_SIZE = 1024;  // Increased for better frequency resolution
const int FFT_SIZE = BUFFER_SIZE ;     // Must match BUFFER_SIZE for proper processing

// Musical scale frequency mapping
const float MIN_AUDIBLE_FREQ = 100.0f;    // 20 Hz (close to lowest piano note)
const float MAX_AUDIBLE_FREQ = 10000.0f; // 20 kHz
const float FREQ_PER_BIN = (float)SAMPLE_RATE / FFT_SIZE; // ~43 Hz per bin
const int MIN_BIN = (int)(MIN_AUDIBLE_FREQ / FREQ_PER_BIN);
const int MAX_BIN = (int)(MAX_AUDIBLE_FREQ / FREQ_PER_BIN);
const int MUSICAL_BINS = 88 * 4; // Fixed number of bins for musical scale

// Audio data
std::vector<float> audioBuffer(BUFFER_SIZE);
std::vector<float> fftMagnitudes(MUSICAL_BINS); // Musical scale distributed frequencies
std::vector<float> rawFFTMagnitudes(FFT_SIZE/2); // Raw FFT data for interpolation
std::vector<float> waveformSamples(MUSICAL_BINS); // Time-domain waveform samples for center-originating visualizer
std::mutex audioMutex;
bool audioRunning = true;  // Changed to true for continuous operation

// Audio and camera textures (declared globally above)

// FFT variables
fftwf_complex *fft_in;
fftwf_complex *fft_out;
fftwf_plan fft_plan;

// PulseAudio
pa_simple *pa_s = NULL;

// Audio processing thread
std::thread audioThread;

// Camera variables
cv::VideoCapture camera;
cv::Mat cameraFrame;
std::mutex cameraMutex;
std::thread cameraThread;
bool cameraRunning = true;

// Shader hot reloading variables
time_t lastShaderModTime = 0;
const char* shaderPath = "cs.cs";

bool hasShaderFileChanged() {
    struct stat fileStat;
    if (stat(shaderPath, &fileStat) == 0) {
        if (fileStat.st_mtime != lastShaderModTime) {
            lastShaderModTime = fileStat.st_mtime;
            return true;
        }
    }
    return false;
}

void processInput(GLFWwindow *window) {
    if(glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

bool initAudio() {
    pa_sample_spec ss;
    int error;

    ss.format = PA_SAMPLE_FLOAT32LE;
    ss.channels = 1;
    ss.rate = SAMPLE_RATE;

    // Use non-blocking mode by setting buffer attributes
    pa_buffer_attr buffer_attr;
    buffer_attr.maxlength = BUFFER_SIZE * sizeof(float) * 4;  // 4x buffer for safety
    buffer_attr.fragsize = BUFFER_SIZE * sizeof(float);       // Fragment size matches our buffer

    pa_s = pa_simple_new(NULL, "Audio Visualizer", PA_STREAM_RECORD, NULL, "Audio input", 
                        &ss, NULL, &buffer_attr, &error);
    if (!pa_s) {
        std::cerr << "Failed to create PulseAudio simple connection: " << pa_strerror(error) << std::endl;
        return false;
    }

    // Initialize FFT
    fft_in = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * FFT_SIZE);
    fft_out = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * FFT_SIZE);
    fft_plan = fftwf_plan_dft_1d(FFT_SIZE, fft_in, fft_out, FFTW_FORWARD, FFTW_ESTIMATE);

    return true;
}

void audioProcessingLoop() {
    while (audioRunning) {
        int error;
        
        // Read audio data
        if (pa_simple_read(pa_s, audioBuffer.data(), BUFFER_SIZE * sizeof(float), &error) < 0) {
            std::cerr << "Failed to read from PulseAudio: " << pa_strerror(error) << std::endl;
            continue;
        }

        // Lock mutex for FFT processing
        std::lock_guard<std::mutex> lock(audioMutex);

        // Apply windowing and copy to FFT input
        for (int i = 0; i < FFT_SIZE; i++) {
            float window = 0.5f * (1.0f - cos(2.0f * M_PI * i / (FFT_SIZE - 1))); // Hanning window
            fft_in[i][0] = audioBuffer[i] * window;
            fft_in[i][1] = 0.0f;
        }

        // Execute FFT
        fftwf_execute(fft_plan);

        // Compute raw FFT magnitudes for all bins
        float lowEnergy = 0.0f;
        float totalEnergy = 0.0f;
        float peakMag = 0.0f;

        for (int i = 0; i < FFT_SIZE/2; i++) {
            float real = fft_out[i][0];
            float imag = fft_out[i][1];
            float mag = sqrt(real * real + imag * imag) / FFT_SIZE;
            rawFFTMagnitudes[i] = mag;
            totalEnergy += mag;
            if (i >= 1 && i <= 8) { // Kick / bass band (~40Hz - 350Hz)
                lowEnergy += mag;
            }
            if (mag > peakMag) {
                peakMag = mag;
            }
        }

        // --- Musical Rhythm Tracking & Auto Speed ---
        static std::vector<float> recentBeatIntervals;
        static float avgLowEnergy = 0.0f;
        static float energyVariance = 0.0f;
        avgLowEnergy = 0.94f * avgLowEnergy + 0.06f * lowEnergy;
        energyVariance = 0.94f * energyVariance + 0.06f * std::abs(lowEnergy - avgLowEnergy);

        static auto lastFrameTime = std::chrono::steady_clock::now();
        auto nowTime = std::chrono::steady_clock::now();
        float dtAudio = std::chrono::duration<float>(nowTime - lastFrameTime).count();
        lastFrameTime = nowTime;
        if (dtAudio <= 0.0f || dtAudio > 0.2f) dtAudio = 0.0232f; // Fallback ~1024/44100s

        static float timeSinceBeat = 0.0f;
        static float estimatedBpm = 120.0f;
        timeSinceBeat += dtAudio;

        // Beat detection: require audible energy, onset exceeding threshold, and refractory window (>= 220ms -> max ~270 BPM)
        if (totalEnergy > 0.002f && lowEnergy > (avgLowEnergy + 1.2f * energyVariance + 0.002f) && timeSinceBeat > 0.22f) {
            float interval = timeSinceBeat;
            timeSinceBeat = 0.0f;

            if (interval >= 0.22f && interval <= 2.5f) {
                recentBeatIntervals.push_back(interval);
                if (recentBeatIntervals.size() > 8) {
                    recentBeatIntervals.erase(recentBeatIntervals.begin());
                }

                std::vector<float> sortedIntervals = recentBeatIntervals;
                std::sort(sortedIntervals.begin(), sortedIntervals.end());
                float medianInterval = sortedIntervals[sortedIntervals.size() / 2];

                // Normalize to standard musical tempo window (70 - 160 BPM)
                while (medianInterval > 0.85f) medianInterval *= 0.5f;
                while (medianInterval < 0.35f) medianInterval *= 2.0f;

                float instantBpm = 60.0f / medianInterval;
                estimatedBpm = 0.95f * estimatedBpm + 0.05f * instantBpm;
                g_radarConfig.detectedBpm.store(estimatedBpm);
            }
        }

        // Determine if musical rhythm is established:
        // 1. Have at least 3 tracked beats in history
        // 2. Consistent interval deviation (< 22% of median interval)
        // 3. Audible energy present (> 0.0025)
        // 4. Beat hasn't gone stale/paused (timeSinceBeat < 1.75 * medianInterval)
        float currentMedian = 60.0f / std::max(estimatedBpm, 40.0f);
        if (recentBeatIntervals.size() >= 3) {
            std::vector<float> sortedIntervals = recentBeatIntervals;
            std::sort(sortedIntervals.begin(), sortedIntervals.end());
            currentMedian = sortedIntervals[sortedIntervals.size() / 2];
            while (currentMedian > 0.85f) currentMedian *= 0.5f;
            while (currentMedian < 0.35f) currentMedian *= 2.0f;
        }

        if (timeSinceBeat > 2.5f && !recentBeatIntervals.empty()) {
            recentBeatIntervals.clear();
        }

        bool regularBeat = false;
        if (recentBeatIntervals.size() >= 3) {
            float devSum = 0.0f;
            for (float iv : recentBeatIntervals) {
                float normIv = iv;
                while (normIv > 1.35f * currentMedian) normIv *= 0.5f;
                while (normIv < 0.65f * currentMedian) normIv *= 2.0f;
                devSum += std::abs(normIv - currentMedian);
            }
            float avgDev = devSum / recentBeatIntervals.size();
            regularBeat = (avgDev < 0.22f * currentMedian);
        }

        bool beatCurrentlyLocked = (recentBeatIntervals.size() >= 3) && regularBeat &&
                                  (totalEnergy > 0.0025f) && (timeSinceBeat < 1.75f * currentMedian);
        static float rhythmConfidence = 0.0f;
        if (beatCurrentlyLocked) {
            rhythmConfidence = std::min(1.0f, rhythmConfidence + 0.10f);
        } else {
            rhythmConfidence = std::max(0.0f, rhythmConfidence - 0.03f);
        }
        bool rhythmEstablished = (rhythmConfidence > 0.5f);
        g_radarConfig.rhythmEstablished.store(rhythmEstablished);

        if (g_radarConfig.autoSpeed.load()) {
            int div = g_radarConfig.rhythmDivision.load();
            float divFactor = 240.0f; // Default: 1 Bar = BPM / 240 rev/s
            if (div == 0) divFactor = 60.0f;        // 1/4 note (1 Beat)
            else if (div == 1) divFactor = 120.0f;  // 1/2 note (2 Beats)
            else if (div == 2) divFactor = 240.0f;  // 1 Bar
            else if (div == 3) divFactor = 480.0f;  // 2 Bars
            else if (div == 4) divFactor = 960.0f;  // 4 Bars

            float targetSpeed = estimatedBpm / divFactor;
            float curSpeed = g_radarConfig.scanningSpeed.load();
            float smoothedSpeed = curSpeed * 0.96f + targetSpeed * 0.04f;
            g_radarConfig.scanningSpeed.store(smoothedSpeed);
        }

        // --- Frequency Range & Scaling Autotuning ---
        float autoGainMultiplier = 1.0f;
        if (g_radarConfig.autoFreq.load()) {
            if (totalEnergy > 0.005f) {
                float cumEnergy = 0.0f;
                int lowBin = 1;
                int highBin = FFT_SIZE / 2 - 1;
                float lowThresh = totalEnergy * 0.03f;
                float highThresh = totalEnergy * 0.92f;

                for (int i = 1; i < FFT_SIZE / 2; i++) {
                    cumEnergy += rawFFTMagnitudes[i];
                    if (cumEnergy < lowThresh) lowBin = i;
                    if (cumEnergy >= highThresh) {
                        highBin = i;
                        break;
                    }
                }

                float detectedMin = std::max(20.0f, lowBin * FREQ_PER_BIN);
                float detectedMax = std::min(18000.0f, highBin * FREQ_PER_BIN);
                if (detectedMax < detectedMin + 300.0f) {
                    detectedMax = detectedMin + 300.0f;
                }

                static float smoothedMinFreq = 100.0f;
                static float smoothedMaxFreq = 10000.0f;
                smoothedMinFreq = 0.98f * smoothedMinFreq + 0.02f * detectedMin;
                smoothedMaxFreq = 0.98f * smoothedMaxFreq + 0.02f * detectedMax;
                g_radarConfig.freqMinHz.store(smoothedMinFreq);
                g_radarConfig.freqMaxHz.store(smoothedMaxFreq);
            }

            // Auto-scaling gain
            static float smoothedPeak = 0.05f;
            smoothedPeak = std::max(smoothedPeak * 0.995f, peakMag);
            autoGainMultiplier = std::min(4.0f, std::max(0.3f, 0.10f / std::max(smoothedPeak, 0.005f)));
            g_radarConfig.autoGain.store(autoGainMultiplier);
        } else {
            g_radarConfig.autoGain.store(1.0f);
        }

        // --- Frequency Scaling (Log / Mel / Linear) ---
        float curMinFreq = g_radarConfig.freqMinHz.load();
        float curMaxFreq = g_radarConfig.freqMaxHz.load();
        if (curMinFreq < 10.0f) curMinFreq = 10.0f;
        if (curMaxFreq <= curMinFreq) curMaxFreq = curMinFreq + 50.0f;

        int scaleMode = g_radarConfig.scaleType.load(); // 0: log, 1: mel, 2: linear

        float logMinFreq = log2(curMinFreq);
        float logMaxFreq = log2(curMaxFreq);
        float logRange = logMaxFreq - logMinFreq;

        float melMinFreq = 2595.0f * log10(1.0f + curMinFreq / 700.0f);
        float melMaxFreq = 2595.0f * log10(1.0f + curMaxFreq / 700.0f);
        float melRange = melMaxFreq - melMinFreq;

        for (int i = 0; i < MUSICAL_BINS; i++) {
            float norm = (float)i / (MUSICAL_BINS - 1);
            float targetFreq = 100.0f;

            if (scaleMode == 1) {
                // Mel scale
                float mel = melMinFreq + melRange * norm;
                targetFreq = 700.0f * (pow(10.0f, mel / 2595.0f) - 1.0f);
            } else if (scaleMode == 2) {
                // Linear scale
                targetFreq = curMinFreq + (curMaxFreq - curMinFreq) * norm;
            } else {
                // Logarithmic scale (default)
                float logFreq = logMinFreq + logRange * norm;
                targetFreq = pow(2.0f, logFreq);
            }

            int targetBin = (int)(targetFreq / FREQ_PER_BIN);
            if (targetBin >= 0 && targetBin < FFT_SIZE / 2) {
                float exactBin = targetFreq / FREQ_PER_BIN;
                float fraction = exactBin - targetBin;
                float val = rawFFTMagnitudes[targetBin];
                if (targetBin + 1 < FFT_SIZE / 2) {
                    val = rawFFTMagnitudes[targetBin] * (1.0f - fraction) + rawFFTMagnitudes[targetBin + 1] * fraction;
                }
                fftMagnitudes[i] = val * autoGainMultiplier;
            } else {
                fftMagnitudes[i] = 0.0f;
            }
        }

        // Resample time-domain audioBuffer (1024) to waveformSamples (352)
        for (int i = 0; i < MUSICAL_BINS; i++) {
            float srcIdx = (float)i * (BUFFER_SIZE - 1) / (MUSICAL_BINS - 1);
            int i0 = (int)srcIdx;
            int i1 = std::min(i0 + 1, BUFFER_SIZE - 1);
            float frac = srcIdx - i0;
            waveformSamples[i] = audioBuffer[i0] * (1.0f - frac) + audioBuffer[i1] * frac;
        }
    }
}

bool initCamera() {
    if (CAMERA_DEV < 0) {
        std::cout << "No camera device specified, using black texture" << std::endl;
        return false; // Will use black texture as fallback
    }
    
    // Try different camera backends
    camera.open(CAMERA_DEV, cv::CAP_V4L2); // Try V4L2 first
    if (!camera.isOpened()) {
        camera.open(CAMERA_DEV); // Try default backend
    }
    
    if (!camera.isOpened()) {
        std::cerr << "Camera device " << CAMERA_DEV << " not available, using black texture" << std::endl;
        return false; // Will use black texture as fallback
    }
    
    // Set camera properties
    camera.set(cv::CAP_PROP_FRAME_WIDTH, TEXTURE_WIDTH);
    camera.set(cv::CAP_PROP_FRAME_HEIGHT, TEXTURE_HEIGHT);
    camera.set(cv::CAP_PROP_FPS, 30);
    
    std::cout << "Camera initialized successfully" << std::endl;
    return true;
}

void cameraProcessingLoop() {
    while (cameraRunning) {
        cv::Mat frame;
        camera >> frame;
        
        if (!frame.empty()) {
            std::lock_guard<std::mutex> lock(cameraMutex);
            cv::resize(frame, cameraFrame, cv::Size(TEXTURE_WIDTH, TEXTURE_HEIGHT));
            cv::flip(cameraFrame, cameraFrame, 0); // Flip vertically for OpenGL
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(16)); // ~60 FPS
    }
}

void cleanupCamera() {
    cameraRunning = false;
    if (cameraThread.joinable()) {
        cameraThread.join();
    }
    
    if (camera.isOpened()) {
        camera.release();
    }
}

void cleanupAudio() {
    audioRunning = false;
    if (audioThread.joinable()) {
        audioThread.join();
    }
    
    if (pa_s) {
        pa_simple_free(pa_s);
        pa_s = NULL;
    }
    
    if (fft_plan) {
        fftwf_destroy_plan(fft_plan);
    }
    if (fft_in) {
        fftwf_free(fft_in);
    }
    if (fft_out) {
        fftwf_free(fft_out);
    }
}

int main(int argc, char* argv[]) {
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--camera" || arg == "-c") {
            if (i + 1 < argc) {
                CAMERA_DEV = std::stoi(argv[i + 1]);
                i++; // Skip the next argument since we've used it
                std::cout << "Using camera device: " << CAMERA_DEV << std::endl;
            } else {
                std::cerr << "Error: --camera requires a device number" << std::endl;
                return -1;
            }
        } else if (arg == "--port" || arg == "-p") {
            if (i + 1 < argc) {
                WEB_PORT = std::stoi(argv[i + 1]);
                i++;
                std::cout << "Using web port: " << WEB_PORT << std::endl;
            } else {
                std::cerr << "Error: --port requires a port number" << std::endl;
                return -1;
            }
        } else if (arg == "--no-web") {
            ENABLE_WEB = false;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [OPTIONS]" << std::endl;
            std::cout << "Options:" << std::endl;
            std::cout << "  -c, --camera <device>   Use camera device number (default: no camera)" << std::endl;
            std::cout << "  -p, --port <port>       Web interface port (default: 8080)" << std::endl;
            std::cout << "  --no-web                Disable web interface" << std::endl;
            std::cout << "  -h, --help              Show this help message" << std::endl;
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            std::cerr << "Use --help for usage information" << std::endl;
            return -1;
        }
    }

    if (ENABLE_WEB) {
        g_webServer = new RadarWebServer(g_radarConfig, WEB_PORT);
        if (!g_webServer->start()) {
            std::cerr << "[Warning] Failed to start web interface on port " << WEB_PORT << std::endl;
        }
    }

    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "COMPUTE", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    
    // Get the actual window resolution
    int windowWidth, windowHeight;
    glfwGetWindowSize(window, &windowWidth, &windowHeight);
    SCR_WIDTH = windowWidth;
    SCR_HEIGHT = windowHeight;
    TEXTURE_WIDTH = SCR_WIDTH;
    TEXTURE_HEIGHT = SCR_HEIGHT;
    glfwMakeContextCurrent(window);
    std::cout << "Resolution: " << SCR_WIDTH << "x" << SCR_HEIGHT << std::endl;

    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSwapInterval(0);

    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // OpenGL info logging (same as before)
    int max_compute_work_group_count[3];
    int max_compute_work_group_size[3];
    int max_compute_work_group_invocations;

    for (int idx = 0; idx < 3; idx++) {
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_COUNT, idx, &max_compute_work_group_count[idx]);
        glGetIntegeri_v(GL_MAX_COMPUTE_WORK_GROUP_SIZE, idx, &max_compute_work_group_size[idx]);
    }

    glGetIntegerv(GL_MAX_COMPUTE_WORK_GROUP_INVOCATIONS, &max_compute_work_group_invocations);

    std::cout << "OpenGL Limitations: " << std::endl;
    std::cout << "maximum number of work groups in X dimension " << max_compute_work_group_count[0] << std::endl;
    std::cout << "maximum number of work groups in Y dimension " << max_compute_work_group_count[1] << std::endl;
    std::cout << "maximum size of a work group in X dimension " << max_compute_work_group_size[0] << std::endl;
    std::cout << "maximum size of a work group in Y dimension " << max_compute_work_group_size[1] << std::endl;
    std::cout << "Number of invocations in a single local work group " << max_compute_work_group_invocations << std::endl;

    Shader screenQuad("screenQuad.vs", "screenQuad.fs");
    screenQuad.use();
    screenQuad.setInt("tex", 0);

    // Create main texture
    glGenTextures(1, &outputTexture);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, outputTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, TEXTURE_WIDTH, TEXTURE_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
    glBindImageTexture(0, outputTexture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);

    // Create audio texture for FFT data and waveform data (row 0 = FFT, row 1 = waveform)
    glGenTextures(1, &audioTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, audioTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, MUSICAL_BINS, 2, 0, GL_RED, GL_FLOAT, NULL);
    glBindImageTexture(1, audioTexture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_R32F);

    // Create camera texture
    glGenTextures(1, &cameraTexture);
    glActiveTexture(GL_TEXTURE2);
    glBindTexture(GL_TEXTURE_2D, cameraTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, TEXTURE_WIDTH, TEXTURE_HEIGHT, 0, GL_BGR, GL_UNSIGNED_BYTE, NULL);
    glBindImageTexture(2, cameraTexture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA8);
    
    // Create backbuffer texture
    glGenTextures(1, &backbufferTexture);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, backbufferTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, TEXTURE_WIDTH, TEXTURE_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
    glBindImageTexture(3, backbufferTexture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);

    // Create storage buffer for simulation states
    const unsigned int SIM_WIDTH = SCR_WIDTH ;
    const unsigned int SIM_HEIGHT = SCR_HEIGHT ;
    const unsigned int N_CHANNELS = 12;
    const size_t stateBufferSize = SIM_WIDTH * SIM_HEIGHT * N_CHANNELS * sizeof(float);
    
    glGenBuffers(1, &stateBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, stateBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, stateBufferSize, NULL, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, stateBuffer);
    
    // Create agent simulation buffers
    const unsigned int AGENT_COUNT = 10000;
    const unsigned int GRID_SIZE = SCR_WIDTH; // Use screen width as grid resolution
    
    // Position buffer (vec2 per agent) - initialize with zeros, let shader handle initialization
    glGenBuffers(1, &positionBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, positionBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, AGENT_COUNT * 2 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, positionBuffer);
    
    // Velocity buffer (vec2 per agent)
    glGenBuffers(1, &velocityBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, velocityBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, AGENT_COUNT * 2 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, velocityBuffer);
    
    // Trail grid buffer (float per grid cell)
    glGenBuffers(1, &trailGridBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, trailGridBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, GRID_SIZE * GRID_SIZE * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, trailGridBuffer);
    
    // Agent grid buffer (vec4 per grid cell)
    glGenBuffers(1, &agentGridBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, agentGridBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, GRID_SIZE * GRID_SIZE * 4 * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, agentGridBuffer);
    
    // Angle buffer (float per agent)
    glGenBuffers(1, &angleBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, angleBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, AGENT_COUNT * sizeof(float), NULL, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, angleBuffer);

    // Create uniform buffer for frame counter and other uniforms
    struct UniformData {
        float kernel[9]; // mat3 = 9 floats
        unsigned int filter_type;
        unsigned int frame;
        float time;
        unsigned int agentCount;
        float padding[3]; // for alignment
    };
    UniformData uniformData = {};
    uniformData.agentCount = 10000; // Number of agents
    
    glGenBuffers(1, &uniformBuffer);
    glBindBuffer(GL_UNIFORM_BUFFER, uniformBuffer);
    glBufferData(GL_UNIFORM_BUFFER, sizeof(UniformData), &uniformData, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_UNIFORM_BUFFER, 4, uniformBuffer);

    // Initialize audio system
    if (!initAudio()) {
        std::cerr << "Failed to initialize audio system" << std::endl;
        return -1;
    }

    // Initialize camera system (optional - continues without camera)
    bool cameraAvailable = initCamera();
    if (!cameraAvailable) {
        std::cout << "Continuing without camera input" << std::endl;
    }

    // Start audio processing thread
    audioThread = std::thread(audioProcessingLoop);
    
    // Start camera processing thread (only if camera is available)
    if (cameraAvailable) {
        cameraThread = std::thread(cameraProcessingLoop);
    }
    
    texturesInitialized = true;

    // Compute shader cache
    ComputeShader* currentComputeShader = nullptr;
    
    // Initialize shader modification time
    hasShaderFileChanged();
    
    // render loop
    int fCounter = 0;
    while (!glfwWindowShouldClose(window)) {
        currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;

        // Cap FPS
        if (deltaTime < FRAME_TIME) {
            float sleepTime = FRAME_TIME - deltaTime;
            glfwWaitEventsTimeout(sleepTime);
            currentFrame = glfwGetTime();
            deltaTime = currentFrame - lastFrame;
        }

        lastFrame = currentFrame;
        processInput(window);

        if(fCounter > 60) {
            float fps = (deltaTime > 0.0f) ? (1.0f / deltaTime) : 0.0f;
            g_radarConfig.currentFps.store(fps);
            g_radarConfig.frameCount.store(frameCounter);
            if (fCounter > 500) {
                std::cout << "FPS: " << fps << std::endl;
                fCounter = 0;
            } else {
                fCounter++;
            }
        } else {
            fCounter++;
        }

        try {
            // Check for shader file changes and reload if necessary
            if (!currentComputeShader || hasShaderFileChanged()) {
                if (currentComputeShader) {
                    std::cout << "Reloading compute shader..." << std::endl;
                    try {
                        ComputeShader* newShader = new ComputeShader("cs.cs");
                        // Only delete old shader if new one compiled successfully
                        delete currentComputeShader;
                        currentComputeShader = newShader;
                        std::cout << "Compute shader reloaded successfully" << std::endl;
                    } catch (const std::exception& e) {
                        std::cout << "Failed to reload compute shader: " << e.what() << std::endl;
                        std::cout << "Keeping previous working version" << std::endl;
                        // Keep using the old shader
                    } catch (...) {
                        std::cout << "Failed to reload compute shader (unknown error) - keeping previous version" << std::endl;
                        // Keep using the old shader
                    }
                } else {
                    currentComputeShader = new ComputeShader("cs.cs");
                    std::cout << "Compute shader loaded successfully" << std::endl;
                }
            }
            
            currentComputeShader->use();
            currentComputeShader->setFloat("time", currentFrame);
            currentComputeShader->setInt("frame", frameCounter);
            
            // Continuous radar phase integration (strictly monotonic forward rotation)
            static float radarPhase = 0.0f;
            float curSpeed = g_radarConfig.scanningSpeed.load();
            radarPhase += curSpeed * deltaTime;
            while (radarPhase >= 1.0f) radarPhase -= 1.0f;
            while (radarPhase < 0.0f) radarPhase += 1.0f;

            // Compute waveform mix based on waveform mode and rhythm established state
            bool isWaveActive = g_radarConfig.isWaveformActive();
            g_radarConfig.waveformActive.store(isWaveActive);

            float targetMix = isWaveActive ? 1.0f : 0.0f;
            static float currentWaveformMix = 1.0f;
            float blendStep = std::min(1.0f, 6.0f * deltaTime);
            currentWaveformMix += (targetMix - currentWaveformMix) * blendStep;

            // Set radar shader controls from web interface
            currentComputeShader->setFloat("scan_phase", radarPhase);
            currentComputeShader->setFloat("scan_speed", curSpeed);
            currentComputeShader->setFloat("decay_time", g_radarConfig.decayTime.load());
            currentComputeShader->setVec2("freq_range", g_radarConfig.freqRangeMin.load(), g_radarConfig.freqRangeMax.load());
            currentComputeShader->setFloat("freq_min_hz", g_radarConfig.freqMinHz.load());
            currentComputeShader->setFloat("freq_max_hz", g_radarConfig.freqMaxHz.load());
            currentComputeShader->setInt("scale_type", g_radarConfig.scaleType.load());
            currentComputeShader->setFloat("nca_perturb", g_radarConfig.ncaPerturb.load());
            currentComputeShader->setFloat("nca_seed", g_radarConfig.ncaSeed.load());
            currentComputeShader->setFloat("nca_step_scale", g_radarConfig.ncaStepScale.load());
            currentComputeShader->setFloat("waveform_mix", currentWaveformMix);
            
            // Update uniform buffer with frame counter and time
            frameCounter++;
            uniformData.frame = frameCounter;
            uniformData.time = currentFrame;
            uniformData.agentCount = 10000;
            glBindBuffer(GL_UNIFORM_BUFFER, uniformBuffer);
            glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(UniformData), &uniformData);

            // Update audio texture with FFT data (row 0) and waveform data (row 1) (with mutex protection)
            {
                std::lock_guard<std::mutex> lock(audioMutex);
                glActiveTexture(GL_TEXTURE1);
                glBindTexture(GL_TEXTURE_2D, audioTexture);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, MUSICAL_BINS, 1, GL_RED, GL_FLOAT, fftMagnitudes.data());
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 1, MUSICAL_BINS, 1, GL_RED, GL_FLOAT, waveformSamples.data());
            }

            // Update camera texture with camera data (with mutex protection)
            {
                std::lock_guard<std::mutex> lock(cameraMutex);
                if (!cameraFrame.empty()) {
                    glActiveTexture(GL_TEXTURE2);
                    glBindTexture(GL_TEXTURE_2D, cameraTexture);
                    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, TEXTURE_WIDTH, TEXTURE_HEIGHT, GL_BGR, GL_UNSIGNED_BYTE, cameraFrame.data);
                }
            }

            glDispatchCompute((unsigned int)TEXTURE_WIDTH/10, (unsigned int)TEXTURE_HEIGHT/10, 1);
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | GL_SHADER_STORAGE_BARRIER_BIT);

            // Render
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            screenQuad.use();
            renderQuad();

            glfwSwapBuffers(window);
            glfwPollEvents();

        } catch (const std::exception& e) {
            std::cout << "Failed to load compute shader: " << e.what() << std::endl;
            // Keep using previous shader if it exists
        } catch (...) {
            std::cout << "Failed to load compute shader (unknown error)" << std::endl;
            // Keep using previous shader if it exists
        }
    }

    // Cleanup
    if (g_webServer) {
        g_webServer->stop();
        delete g_webServer;
        g_webServer = nullptr;
    }
    cleanupAudio();
    cleanupCamera();
    
    if (currentComputeShader) {
        delete currentComputeShader;
    }

    glfwTerminate();
    return EXIT_SUCCESS;
}

unsigned int quadVAO = 0;
unsigned int quadVBO;
void renderQuad() {
    if (quadVAO == 0) {
        float quadVertices[] = {
            -1.0f,  1.0f, 0.0f, 0.0f, 1.0f,
            -1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
             1.0f,  1.0f, 0.0f, 1.0f, 1.0f,
             1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
        };
        
        glGenVertexArrays(1, &quadVAO);
        glGenBuffers(1, &quadVBO);
        glBindVertexArray(quadVAO);
        glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
        glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    }
    glBindVertexArray(quadVAO);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    glBindVertexArray(0);
}
