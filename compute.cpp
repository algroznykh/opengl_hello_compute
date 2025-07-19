#include <iostream>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <pulse/pulseaudio.h>
#include <pulse/simple.h>
#include <pulse/error.h>
#include <thread>
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

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    glViewport(0, 0, width, height);
}

void renderQuad();

const unsigned int SCR_WIDTH = 2560;
const unsigned int SCR_HEIGHT = 1600;
const unsigned int TEXTURE_WIDTH = 1000, TEXTURE_HEIGHT = 1000;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;
const float FPS_CAP = 90.0f;
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
std::mutex audioMutex;
bool audioRunning = true;  // Changed to true for continuous operation

// Audio texture
unsigned int audioTexture;

// Camera texture
unsigned int cameraTexture;

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
        for (int i = 0; i < FFT_SIZE/2; i++) {
            float real = fft_out[i][0];
            float imag = fft_out[i][1];
            rawFFTMagnitudes[i] = sqrt(real * real + imag * imag) / FFT_SIZE;
        }

        // Map to logarithmic frequency scale for even overtone spacing
        for (int i = 0; i < MUSICAL_BINS; i++) {
            // Logarithmic frequency distribution from MIN_AUDIBLE_FREQ to MAX_AUDIBLE_FREQ
            float logMinFreq = log2(MIN_AUDIBLE_FREQ);
            float logMaxFreq = log2(MAX_AUDIBLE_FREQ);
            float logRange = logMaxFreq - logMinFreq;
            
            // Map bin index to logarithmic frequency
            float logFreq = logMinFreq + (logRange * i) / (MUSICAL_BINS - 1);
            float targetFreq = pow(2, logFreq);
            int targetBin = (int)(targetFreq / FREQ_PER_BIN);
            
            // Clamp to valid range and interpolate if needed
            if (targetBin >= 0 && targetBin < FFT_SIZE/2) {
                fftMagnitudes[i] = rawFFTMagnitudes[targetBin];
                
                // Linear interpolation for smoother transitions
                if (targetBin + 1 < FFT_SIZE/2) {
                    float exactBin = targetFreq / FREQ_PER_BIN;
                    float fraction = exactBin - targetBin;
                    fftMagnitudes[i] = rawFFTMagnitudes[targetBin] * (1.0f - fraction) + 
                                     rawFFTMagnitudes[targetBin + 1] * fraction;
                }
                
                // Apply logarithmic scaling for better visualization
                //fftMagnitudes[i] = fftMagnitudes[i] > 0.0f ? log10(1.0f + fftMagnitudes[i] * 10.0f) : 0.0f;
            } else {
                fftMagnitudes[i] = 0.0f;
            }
        }
    }
}

bool initCamera() {
    // Try different camera backends
    camera.open(2, cv::CAP_V4L2); // Try V4L2 first
    if (!camera.isOpened()) {
        camera.open(2); // Try default backend
    }
    
    if (!camera.isOpened()) {
        std::cerr << "No camera available, using black texture" << std::endl;
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

int main() {
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "COMPUTE", NULL, NULL);
    if (window == NULL) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);

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
    unsigned int texture;
    glGenTextures(1, &texture);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, TEXTURE_WIDTH, TEXTURE_HEIGHT, 0, GL_RGBA, GL_FLOAT, NULL);
    glBindImageTexture(0, texture, 0, GL_FALSE, 0, GL_READ_WRITE, GL_RGBA32F);

    // Create audio texture for FFT data
    glGenTextures(1, &audioTexture);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, audioTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, MUSICAL_BINS, 1, 0, GL_RED, GL_FLOAT, NULL);
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
    unsigned int backbufferTexture;
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
    const unsigned int SIM_WIDTH = SCR_WIDTH;
    const unsigned int SIM_HEIGHT = SCR_HEIGHT;
    const unsigned int N_CHANNELS = 12;
    const size_t stateBufferSize = SIM_WIDTH * SIM_HEIGHT * N_CHANNELS * sizeof(float);
    
    unsigned int stateBuffer;
    glGenBuffers(1, &stateBuffer);
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, stateBuffer);
    glBufferData(GL_SHADER_STORAGE_BUFFER, stateBufferSize, NULL, GL_DYNAMIC_DRAW);
    glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, stateBuffer);

    // Create uniform buffer for frame counter and other uniforms
    struct UniformData {
        float kernel[9]; // mat3 = 9 floats
        unsigned int filter_type;
        unsigned int frame;
        float padding[2]; // for alignment
    };
    UniformData uniformData = {};
    
    unsigned int uniformBuffer;
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

    // Compute shader cache
    ComputeShader* currentComputeShader = nullptr;
    
    // Initialize shader modification time
    hasShaderFileChanged();
    
    // render loop
    int fCounter = 0;
    unsigned int frameCounter = 0;
    while (!glfwWindowShouldClose(window)) {
        float currentFrame = glfwGetTime();
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

        if(fCounter > 500) {
            std::cout << "FPS: " << 1 / deltaTime << std::endl;
            fCounter = 0;
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
            currentComputeShader->setFloat("t", currentFrame);
            
            // Update uniform buffer with frame counter
            frameCounter++;
            uniformData.frame = frameCounter;
            glBindBuffer(GL_UNIFORM_BUFFER, uniformBuffer);
            glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(UniformData), &uniformData);

            // Update audio texture with FFT data (with mutex protection)
            {
                std::lock_guard<std::mutex> lock(audioMutex);
                glActiveTexture(GL_TEXTURE1);
                glBindTexture(GL_TEXTURE_2D, audioTexture);
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, MUSICAL_BINS, 1, GL_RED, GL_FLOAT, fftMagnitudes.data());
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
            glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT);

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
