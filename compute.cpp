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

#include <random>
#include "shader_c.h"
#include "shader_m.h"
#include "web_server.h"
#include "font8x8.h"

// Global radar configuration and web server
RadarConfig g_radarConfig;
RadarWebServer* g_webServer = nullptr;
int WEB_PORT = 8080;
bool ENABLE_WEB = true;

// Global texture and buffer handles for recreation
unsigned int outputTexture, audioTexture, cameraTexture, backbufferTexture;
unsigned int stateBuffer, positionBuffer, velocityBuffer, trailGridBuffer, agentGridBuffer, angleBuffer;
unsigned int uniformBuffer, ncaWeightBuffer;
unsigned int SCR_WIDTH = 800, SCR_HEIGHT = 600;
unsigned int TEXTURE_WIDTH = 800, TEXTURE_HEIGHT = 600;
bool texturesInitialized = false;

// Original trained NCA weights baseline
const float ORIGINAL_NCA_B[12] = {
    -332.0f, -212.0f, -353.0f, 566.0f, -86.0f, 837.0f, 510.0f, -339.0f, 373.0f, 414.0f, -787.0f, -712.0f
};

const float ORIGINAL_NCA_W[48][12] = {
    { -816, 142, -40, 142, -119, 0, 758, 70, 99, -149, -165, -141 },
    { -24, -1182, -67, 438, 267, 190, 358, 113, 191, -64, -196, -263 },
    { 10, 170, -785, -49, 98, -39, 733, -75, 210, -161, -112, -107 },
    { 60, -183, 158, -966, 352, -336, 588, -356, 123, -531, -364, -201 },
    { -11, -89, -77, -259, -2788, 276, 46, 56, 72, -28, -179, 66 },
    { 24, -43, -1, 463, -452, -2262, 95, 59, -51, 3, -222, -216 },
    { -162, 158, -113, 19, -26, -66, -1856, 49, -27, -302, -99, -366 },
    { 58, 9, 203, -20, -200, -76, -28, -2600, -206, 29, 127, 358 },
    { -68, -222, -143, -254, -287, 157, 255, 400, -3107, -87, 373, -420 },
    { -59, -75, 1, -464, 19, 82, -26, 181, -187, -1802, 124, -557 },
    { 191, 166, 218, -398, 22, 131, 136, 552, 301, -273, -1490, -123 },
    { 188, 142, 283, -677, 143, 84, -154, -80, 212, -34, -69, -1619 },
    { 257, -95, -4, 62, -3, 36, 29, -17, 16, 12, 34, 6 },
    { -79, 218, -137, -68, -23, -3, 63, -20, -12, -14, -1, 16 },
    { -97, -42, 211, 91, -3, 26, 42, 12, -13, -40, -5, 37 },
    { -106, -5, -160, 673, -500, 177, 219, 41, -167, -479, -273, -478 },
    { -137, -144, -138, -30, 402, -100, 20, 59, -57, -11, -16, 93 },
    { 144, 174, 145, 84, 14, 128, -180, -234, -127, -221, 59, 192 },
    { 8, 4, 6, 76, -92, -343, 1, -6, 97, -20, -50, -41 },
    { -38, -38, -24, 42, 158, -184, -28, -130, 70, -48, 28, -43 },
    { -178, -190, -164, -131, 116, -15, -154, -93, 229, 57, 33, 193 },
    { -52, -84, -60, -126, 86, -50, 32, 210, -71, 268, 123, -203 },
    { 213, 184, 191, -65, 162, -25, 128, 445, 80, 69, 432, -191 },
    { -124, -117, -125, -59, -130, -84, -42, 173, 261, 14, 5, -278 },
    { 51, -269, -169, -412, 300, -62, 148, -72, 219, -66, 335, 492 },
    { -157, 83, -128, -397, 144, -98, 369, 17, 183, -123, 392, 490 },
    { -193, -370, 35, -306, 228, -5, 155, 49, 169, -15, 384, 480 },
    { 198, 172, 203, -172, 179, 92, 127, 183, 450, -2, -66, 222 },
    { 48, 39, 62, 169, 1160, -98, 174, 32, -15, 100, 117, 65 },
    { -36, -63, -54, 279, -132, -1232, -7, 130, 90, 37, -20, -38 },
    { 14, 54, 46, -129, 34, 85, -480, 100, -83, -325, -78, 107 },
    { -228, -276, -217, -255, -117, -148, 0, 1, -156, 257, 164, -8 },
    { -27, -95, -41, -34, -35, 33, 150, 269, -1215, -116, 299, -217 },
    { -294, -409, -300, -267, 104, -40, 203, 101, 133, -150, 368, 8 },
    { 139, 154, 113, 27, 19, 274, 103, 150, 217, 174, -293, 300 },
    { 82, 39, 56, -211, -121, -44, -61, 557, -982, -167, 358, -339 },
    { -32, 29, 20, -6, -28, -19, 21, 1, 0, 21, -23, -2 },
    { 13, -12, 38, 28, -47, 32, -42, 38, -37, -14, -27, -65 },
    { 43, 5, -36, -21, 10, -21, 9, 15, -34, -19, -16, -30 },
    { 528, 533, 528, -52, -199, -120, -6, 140, 211, -186, 88, 29 },
    { 25, 9, 27, -71, 30, -8, -6, 73, 31, 44, 213, -1 },
    { 227, 224, 226, 70, 71, -43, 101, 100, 168, -405, 69, -83 },
    { -142, -184, -148, -169, -115, 119, 25, 26, 266, 296, 296, 218 },
    { 118, 164, 96, 279, -120, 98, 41, 30, 233, -190, -124, 204 },
    { -73, -95, -63, -89, 65, -13, -5, -50, 77, -51, 38, 87 },
    { -55, -71, -42, -44, 29, 6, 38, -174, 77, 5, -45, 112 },
    { 148, 73, 136, -81, -113, 12, 226, 133, -417, -305, 226, -81 },
    { 100, 152, 115, 28, -229, 26, -272, -200, 528, -81, -102, 235 }
};

struct NcaGpuGenome {
    float W[576]; // 48 x 12
    float B[12];  // 12
};

class NcaGeneticEngine {
public:
    NcaGpuGenome baseline;
    NcaGpuGenome candidate;
    std::vector<NcaGpuGenome> history;
    int generation = 0;
    bool hasCandidate = false;
    int mutatedCount = 0;
    bool dirty = false;
    std::mt19937 rng;

    NcaGeneticEngine() : rng(1337) {
        resetToOriginal();
    }

    void resetToOriginal() {
        for (int i = 0; i < 48; i++) {
            for (int c = 0; c < 12; c++) {
                baseline.W[i * 12 + c] = ORIGINAL_NCA_W[i][c];
            }
        }
        for (int c = 0; c < 12; c++) {
            baseline.B[c] = ORIGINAL_NCA_B[c];
        }
        candidate = baseline;
        history.clear();
        generation = 0;
        hasCandidate = false;
        mutatedCount = 0;
        dirty = true;
    }

    int mutate(const std::string& target, float scale, float rate) {
        candidate = baseline;
        std::uniform_real_distribution<float> probDist(0.0f, 1.0f);
        std::normal_distribution<float> normDist(0.0f, 1.0f);

        int count = 0;
        float effScale = std::max(0.01f, std::min(1.0f, scale));
        float effRate = std::max(0.01f, std::min(1.0f, rate));
        bool isEverything = (target == "everything" || target == "all_100");

        // Rows in W:
        // 0..11: xs (state recurrence)
        // 12..23: ps (spatial perception: Sobel / Lap)
        // 24..35: abs(xs) (nonlinear state)
        // 36..47: abs(ps) (nonlinear perception)

        for (int row = 0; row < 48; row++) {
            bool targetRow = false;
            if (isEverything || target == "all") {
                targetRow = true;
            } else if (target == "perception") {
                targetRow = (row >= 12 && row < 24) || (row >= 36 && row < 48);
            } else if (target == "state") {
                targetRow = (row < 12) || (row >= 24 && row < 36);
            } else if (target == "nonlinear") {
                targetRow = (row >= 24 && row < 48);
            }

            if (!targetRow) continue;

            for (int col = 0; col < 12; col++) {
                if (isEverything || probDist(rng) < effRate) {
                    int idx = row * 12 + col;
                    float orig = candidate.W[idx];
                    float delta = normDist(rng) * (std::abs(orig) * effScale + 40.0f * effScale);
                    candidate.W[idx] += delta;
                    count++;
                }
            }
        }

        if (isEverything || target == "all" || target == "biases" || target == "bias") {
            for (int c = 0; c < 12; c++) {
                if (isEverything || probDist(rng) < effRate) {
                    float orig = candidate.B[c];
                    float delta = normDist(rng) * (std::abs(orig) * effScale + 40.0f * effScale);
                    candidate.B[c] += delta;
                    count++;
                }
            }
        }

        mutatedCount = count;
        hasCandidate = true;
        dirty = true;
        return count;
    }

    void cancel() {
        if (hasCandidate) {
            candidate = baseline;
            hasCandidate = false;
            mutatedCount = 0;
            dirty = true;
        }
    }

    void reinforce() {
        if (hasCandidate) {
            history.push_back(baseline);
            if (history.size() > 25) {
                history.erase(history.begin());
            }
            baseline = candidate;
            generation++;
            hasCandidate = false;
            mutatedCount = 0;
            dirty = true;
        }
    }

    bool rollback() {
        if (!history.empty()) {
            baseline = history.back();
            history.pop_back();
            candidate = baseline;
            if (generation > 0) generation--;
            hasCandidate = false;
            mutatedCount = 0;
            dirty = true;
            return true;
        }
        return false;
    }

    const NcaGpuGenome& getActiveGenome() const {
        return hasCandidate ? candidate : baseline;
    }
};

NcaGeneticEngine g_geneticEngine;

void imprintTextMessage(const std::string& msg) {
    if (msg.empty() || TEXTURE_WIDTH == 0 || TEXTURE_HEIGHT == 0) return;

    std::string cleanMsg;
    for (char ch : msg) {
        if (ch >= 32 && ch <= 126) {
            cleanMsg += ch;
        } else if (ch == '\n' || ch == '\t') {
            cleanMsg += ' ';
        }
    }
    if (cleanMsg.empty()) cleanMsg = "BYOB";
    if (cleanMsg.length() > 40) cleanMsg = cleanMsg.substr(0, 40);

    // Smart line splitting: test if 2 lines gives even bigger glyphs for phrases with spaces
    std::vector<std::string> lines;
    size_t spacePos = cleanMsg.find(' ');
    if (cleanMsg.length() > 5 && spacePos != std::string::npos && spacePos > 1 && spacePos < cleanMsg.length() - 2) {
        std::string l1 = cleanMsg.substr(0, spacePos);
        std::string l2 = cleanMsg.substr(spacePos + 1);
        int u1 = (int)cleanMsg.length() * 8 + (int)(cleanMsg.length() - 1) * 2;
        int s1 = std::min((int)((TEXTURE_WIDTH * 0.95f) / std::max(1, u1)), (int)((TEXTURE_HEIGHT * 0.88f) / 8));

        int maxLu = std::max((int)l1.length(), (int)l2.length()) * 8 + (std::max((int)l1.length(), (int)l2.length()) - 1) * 2;
        int hUnits = 8 * 2 + 3; // 2 lines
        int s2 = std::min((int)((TEXTURE_WIDTH * 0.95f) / std::max(1, maxLu)), (int)((TEXTURE_HEIGHT * 0.88f) / hUnits));

        if (s2 > s1) {
            lines.push_back(l1);
            lines.push_back(l2);
        } else {
            lines.push_back(cleanMsg);
        }
    } else {
        lines.push_back(cleanMsg);
    }

    int maxLineChars = 0;
    for (const auto& l : lines) {
        if ((int)l.length() > maxLineChars) maxLineChars = (int)l.length();
    }

    const int charUnits = 8;
    const int spacingUnits = 2;
    int lineUnitsW = maxLineChars * charUnits + (maxLineChars - 1) * spacingUnits;
    int numLines = (int)lines.size();
    int lineUnitsH = numLines * charUnits + (numLines - 1) * 3;

    // Maximum charScale that fills 95% width and 90% height
    int maxScaleW = (int)((TEXTURE_WIDTH * 0.95f) / std::max(1, lineUnitsW));
    int maxScaleH = (int)((TEXTURE_HEIGHT * 0.90f) / std::max(1, lineUnitsH));
    int charScale = std::max(3, std::min(maxScaleW, maxScaleH));

    int charWidth = charUnits * charScale;
    int charHeight = charUnits * charScale;
    int charSpacing = spacingUnits * charScale;
    int lineSpacing = 3 * charScale;

    int totalH = numLines * charHeight + (numLines - 1) * lineSpacing;
    int startY = (TEXTURE_HEIGHT - totalH) / 2;
    if (startY < 0) startY = 0;

    // Clear entire backbuffer texture so previous smaller imprint doesn't ghost
    std::vector<float> emptyBack(TEXTURE_WIDTH * TEXTURE_HEIGHT * 4, 0.0f);
    glActiveTexture(GL_TEXTURE3);
    glBindTexture(GL_TEXTURE_2D, backbufferTexture);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, TEXTURE_WIDTH, TEXTURE_HEIGHT, GL_RGBA, GL_FLOAT, emptyBack.data());

    for (int li = 0; li < numLines; li++) {
        const std::string& lineText = lines[li];
        int lineLen = (int)lineText.length();
        int lineTotalW = lineLen * charWidth + (lineLen - 1) * charSpacing;
        int startX = (TEXTURE_WIDTH - lineTotalW) / 2;
        if (startX < 0) startX = 0;
        if (startX + lineTotalW > (int)TEXTURE_WIDTH) lineTotalW = TEXTURE_WIDTH - startX;

        int lineY = startY + (numLines - 1 - li) * (charHeight + lineSpacing);
        int clampedH = std::min(charHeight, (int)TEXTURE_HEIGHT - lineY);
        if (clampedH <= 0) continue;

        std::vector<float> patch(lineTotalW * clampedH * 4, 0.0f);

        for (size_t ci = 0; ci < lineText.length(); ci++) {
            uint8_t ascii = (uint8_t)lineText[ci];
            int glyphIdx = ascii - 32;
            if (glyphIdx < 0 || glyphIdx >= 95) continue;
            int glyphStartX = (int)ci * (charWidth + charSpacing);

            for (int r = 0; r < 8; r++) {
                uint8_t rowBits = FONT8X8[glyphIdx][r];
                int flippedR = 7 - r;

                for (int c = 0; c < 8; c++) {
                    if ((rowBits >> (7 - c)) & 1) {
                        for (int dy = 0; dy < charScale; dy++) {
                            int py = flippedR * charScale + dy;
                            if (py >= clampedH) continue;

                            for (int dx = 0; dx < charScale; dx++) {
                                int px = glyphStartX + c * charScale + dx;
                                if (px >= lineTotalW) continue;

                                int pixelIdx = (py * lineTotalW + px) * 4;
                                patch[pixelIdx + 0] = 2.4f;
                                patch[pixelIdx + 1] = 0.6f;
                                patch[pixelIdx + 2] = 3.0f;
                                patch[pixelIdx + 3] = 1.0f;
                            }
                        }
                    }
                }
            }
        }

        glTexSubImage2D(GL_TEXTURE_2D, 0, startX, lineY, lineTotalW, clampedH, GL_RGBA, GL_FLOAT, patch.data());
    }

    std::cout << "[Easter Egg] Giant imprint: \"" << cleanMsg << "\" (charScale=" << charScale 
              << ", lines=" << numLines << ")" << std::endl;
}

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

    // Dynamic NCA genome weights buffer (binding 11) for Human-Assisted Genetic Algorithm
    glGenBuffers(1, &ncaWeightBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ncaWeightBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(NcaGpuGenome), &g_geneticEngine.getActiveGenome(), GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, ncaWeightBuffer);

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

            // Handle pending Easter egg message imprints
            std::string pendingMsg;
            if (g_radarConfig.popPendingImprint(pendingMsg)) {
                imprintTextMessage(pendingMsg);
            }

            // Handle pending genetic algorithm actions
            RadarConfig::GeneticRequest genReq;
            if (g_radarConfig.popPendingGenetic(genReq)) {
                if (genReq.action == "mutate") {
                    g_geneticEngine.mutate(genReq.target, genReq.scale, genReq.rate);
                } else if (genReq.action == "mutate_everything") {
                    g_geneticEngine.mutate("everything", genReq.scale, 1.0f);
                } else if (genReq.action == "cancel" || genReq.action == "revert") {
                    g_geneticEngine.cancel();
                } else if (genReq.action == "reinforce" || genReq.action == "commit" || genReq.action == "keep") {
                    g_geneticEngine.reinforce();
                } else if (genReq.action == "rollback") {
                    g_geneticEngine.rollback();
                } else if (genReq.action == "reset") {
                    g_geneticEngine.resetToOriginal();
                }
            }
            g_radarConfig.ncaGeneration.store(g_geneticEngine.generation);
            g_radarConfig.ncaHasCandidate.store(g_geneticEngine.hasCandidate);
            g_radarConfig.ncaMutatedCount.store(g_geneticEngine.mutatedCount);

            if (g_geneticEngine.dirty) {
                glBindBuffer(GL_SHADER_STORAGE_BUFFER, ncaWeightBuffer);
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(NcaGpuGenome), &g_geneticEngine.getActiveGenome());
                g_geneticEngine.dirty = false;
            }

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
            currentComputeShader->setFloat("nca_topology", g_radarConfig.ncaTopology.load());
            currentComputeShader->setFloat("nca_pole_radial", g_radarConfig.ncaPoleRadial.load());
            currentComputeShader->setFloat("nca_pole_azimuthal", g_radarConfig.ncaPoleAzimuthal.load());
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
