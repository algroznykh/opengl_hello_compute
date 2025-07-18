#version 430 core

layout (local_size_x = 10, local_size_y = 10, local_size_z = 1) in;

// ----------------------------------------------------------------------------
//
// uniforms
//
// ----------------------------------------------------------------------------

layout(rgba32f, binding = 0) uniform image2D imgOutput;
layout(r32f, binding = 1) uniform image2D audioTexture;

layout (location = 0) uniform float time;                 /** Time */

// ----------------------------------------------------------------------------
//
// functions
//
// ----------------------------------------------------------------------------

float circle(vec2 p, float r) {
    return length(p) - pow(r, 2.);
    }

void main() {
	vec4 value = vec4(0.0, 0.0, 0.0, 1.0);
	ivec2 texelCoord = ivec2(gl_GlobalInvocationID.xy);
    
    vec2 uv = vec2(texelCoord.xy) / (vec2(gl_NumWorkGroups.xy) * vec2(gl_WorkGroupSize.xy));

    // Get audio data - map X coordinate to frequency bin
    int audioIndex = int(uv.x * 256.0); // 256 = FFT_SIZE/2
    audioIndex = clamp(audioIndex, 0, 255);
    
    float audioMagnitude = imageLoad(audioTexture, ivec2(audioIndex, 0)).r;
    audioMagnitude = clamp(audioMagnitude * 200.0, 0.0, 1.0); // Amplify and clamp
    
    // Create spectrum bars across the full width
    float barHeight = audioMagnitude;
    
    // Frequency-based color mapping
    vec3 color = vec3(0.0);
    float freqRatio = float(audioIndex) / 256.0;
    
    // Rainbow spectrum: red -> orange -> yellow -> green -> blue -> purple
    if (freqRatio < 0.16) {
        color = mix(vec3(1.0, 0.0, 0.0), vec3(1.0, 0.5, 0.0), freqRatio * 6.0); // Red to orange
    } else if (freqRatio < 0.33) {
        color = mix(vec3(1.0, 0.5, 0.0), vec3(1.0, 1.0, 0.0), (freqRatio - 0.16) * 6.0); // Orange to yellow
    } else if (freqRatio < 0.50) {
        color = mix(vec3(1.0, 1.0, 0.0), vec3(0.0, 1.0, 0.0), (freqRatio - 0.33) * 6.0); // Yellow to green
    } else if (freqRatio < 0.66) {
        color = mix(vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0), (freqRatio - 0.50) * 6.0); // Green to blue
    } else if (freqRatio < 0.83) {
        color = mix(vec3(0.0, 0.0, 1.0), vec3(0.5, 0.0, 1.0), (freqRatio - 0.66) * 6.0); // Blue to purple
    } else {
        color = mix(vec3(0.5, 0.0, 1.0), vec3(1.0, 0.0, 1.0), (freqRatio - 0.83) * 6.0); // Purple to magenta
    }
    
    // Draw spectrum bars from bottom up
    float intensity = 0.0;
    if (uv.y < barHeight) {
        intensity = 1.0 - (uv.y / barHeight) * 0.3; // Fade towards top
    }
    
    // Add some glow effect
    float glowDistance = abs(uv.y - barHeight);
    if (glowDistance < 0.05) {
        intensity += (1.0 - glowDistance / 0.05) * 0.3;
    }
    
    vec3 finalColor = color * intensity;
    
    // Add a subtle background grid
    float gridX = mod(uv.x * 64.0, 1.0);
    float gridY = mod(uv.y * 32.0, 1.0);
    if (gridX < 0.05 || gridY < 0.05) {
        finalColor += vec3(0.02);
    }
    
    value = vec4(finalColor, 1.0);
	imageStore(imgOutput, texelCoord, value);
}

