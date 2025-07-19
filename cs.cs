#version 430 core

layout (local_size_x = 10, local_size_y = 10, local_size_z = 1) in;

// ----------------------------------------------------------------------------
//
// uniforms
//
// ----------------------------------------------------------------------------

layout(rgba32f, binding = 0) uniform image2D imgOutput;
layout(r32f, binding = 1) uniform image2D audioTexture;
layout(rgba8, binding = 2) uniform image2D cameraInput;
layout(rgba32f, binding = 3) uniform image2D backbuffer;

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

    // Sample camera input
    vec4 cameraColor = imageLoad(cameraInput, texelCoord);

    // Get audio data - map Y coordinate to musical scale bins
    // Musical scale: logarithmic distribution from 20Hz to 20kHz
    float MUSICAL_BINS = 88.0 * 4.; // Fixed number of musical scale bins
    int audioIndex = int(uv.y * MUSICAL_BINS);
    audioIndex = clamp(audioIndex, 0, int(MUSICAL_BINS)-1);
    
    int SPECTRUM_SHIFT = 0;
    float audioMagnitude = imageLoad(audioTexture, ivec2(audioIndex + SPECTRUM_SHIFT, 0)).r;
    audioMagnitude = clamp(audioMagnitude * 200.0, 0.0, 1.0); // Amplify and clamp
    
    vec3 color = vec3(audioMagnitude);
    float intensity = floor(uv.x + .001);
    vec3 spectrum = color * intensity;
    vec3 finalColor = spectrum;
    
    // Add a subtle background grid
    float gridX = mod(uv.x * 64.0, 1.0);
    float gridY = mod(uv.y * 32.0, 1.0);
    if (gridX < 0.05 || gridY < 0.05) {
        finalColor += vec3(0.02);
    }
    
    // Blend with camera input
    finalColor = mix(finalColor, cameraColor.rgb, 0.3);

    vec4 back = imageLoad(backbuffer, texelCoord + ivec2(1, 0));
    
    value = vec4(finalColor, 1.0) + back;
    back += vec4(spectrum, 1.);
    imageStore(backbuffer, texelCoord, back);
	imageStore(imgOutput, texelCoord, value);
}

