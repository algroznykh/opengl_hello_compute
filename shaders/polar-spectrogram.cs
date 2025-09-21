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


vec4 sampleBilinear(image2D img, vec2 uv) {
    vec2 texSize = vec2(gl_NumWorkGroups.xy * gl_WorkGroupSize.xy);
    vec2 pixelCoord = uv * texSize - 0.5;
    vec2 floorCoord = floor(pixelCoord);
    vec2 fracCoord = pixelCoord - floorCoord;
    
    ivec2 i00 = ivec2(floorCoord);
    ivec2 i10 = i00 + ivec2(1, 0);
    ivec2 i01 = i00 + ivec2(0, 1);
    ivec2 i11 = i00 + ivec2(1, 1);
    
    // Handle wrapping or clamping
    i00 = clamp(i00, ivec2(0), ivec2(texSize) - 1);
    i10 = clamp(i10, ivec2(0), ivec2(texSize) - 1);
    i01 = clamp(i01, ivec2(0), ivec2(texSize) - 1);
    i11 = clamp(i11, ivec2(0), ivec2(texSize) - 1);
    
    vec4 s00 = imageLoad(img, i00);
    vec4 s10 = imageLoad(img, i10);
    vec4 s01 = imageLoad(img, i01);
    vec4 s11 = imageLoad(img, i11);
    
    vec4 s0 = mix(s00, s10, fracCoord.x);
    vec4 s1 = mix(s01, s11, fracCoord.x);
    
    return mix(s0, s1, fracCoord.y);
}

vec2 cart(vec2 polar) {
    float angle = polar.y * (2.0 * 3.14159) - 3.14159;  
    vec2 centered;
    centered.x = polar.x * cos(angle);
    centered.y = polar.x * sin(angle);

    // Scale back from [-1,1] to [-0.5,0.5] and center at 0.5
    centered /= 2.0;
    vec2 cart = centered + 0.5;
    return cart;
}

void main() {
	vec4 value = vec4(0.0, 0.0, 0.0, 1.0);
	ivec2 texelCoord = ivec2(gl_GlobalInvocationID.xy);
    
    ivec2 imgSize = imageSize(imgOutput);  // or imageSize(backbuffer)
    vec2 uv = vec2(texelCoord.xy) / vec2(imgSize);
    //vec2 uv = vec2(texelCoord.xy) / (vec2(gl_NumWorkGroups.xy) * vec2(gl_WorkGroupSize.xy));
    
    // Convert to polar coordinates - center at (0.5, 0.5)
    vec2 centered = uv - 0.5;
    centered *= 2.;
    float radius = length(centered);
    float angle = atan(centered.y, centered.x);
    
    // Normalize angle to [0, 1] range
    angle = (angle + 3.14159) / (2.0 * 3.14159);
    angle = 1. - angle;
    angle = angle + .75;
    angle = fract(angle);

    vec2 polar = vec2(radius, angle);

    // Sample camera input
    vec4 cameraColor = imageLoad(cameraInput, texelCoord);

    // Get audio data - map angle to musical scale bins for circular spectrogram
    float MUSICAL_BINS = 88.0 * 4.; // Fixed number of musical scale bins
    //int audioIndex = int((1.-radius) * MUSICAL_BINS);
    int audioIndex = int((1.-uv.y) * MUSICAL_BINS);
    audioIndex = clamp(audioIndex, 0, int(MUSICAL_BINS)-1);
    
    int SPECTRUM_SHIFT = 0;
    float audioMagnitude = imageLoad(audioTexture, ivec2(audioIndex + SPECTRUM_SHIFT, 0)).r;
    audioMagnitude = clamp(audioMagnitude * 200.0, 0.0, 1.0); // Amplify and clamp
    
    vec3 color = vec3(audioMagnitude);
    // Use radius for intensity instead of x coordinate
    //float intensity = angle < 0.01 ? 1. : 0.;
    float intensity = uv.x > .99 ? 1. : 0.;
    vec3 spectrum = color * intensity;
    vec3 finalColor ;
    
    // Add radial grid lines
    float gridAngle = mod(angle * 32.0, 1.0);
    float gridRadius = mod(radius * 16.0, 1.0);
    if (gridAngle < 0.05 || gridRadius < 0.05) {
        finalColor += vec3(0.02);
    }
    
    // Blend with camera input
    finalColor = mix(finalColor, cameraColor.rgb, 0.3);

    // Spinning disc effect - rotate backbuffer sampling based on time
    float q = 0.001;
    //vec2 rotatedUV = mat2(cos(q), -sin(q), sin(q), cos(q)) * centered + .5;
    vec2 rotatedUV = cart(polar + vec2(0., q))  ;

    // Convert back to texel coordinates for backbuffer sampling
    ivec2 rotatedTexel = ivec2(rotatedUV * vec2(imgSize));
    //rotatedTexel = clamp(rotatedTexel, ivec2(0), imgSize - 1);
    //ivec2 rotatedTexel = ivec2(rotatedUV * vec2(gl_NumWorkGroups.xy) * vec2(gl_WorkGroupSize.xy));
    //rotatedTexel = clamp(rotatedTexel, ivec2(0), ivec2(gl_NumWorkGroups.xy * gl_WorkGroupSize.xy) - 1);
    
    //vec4 back = imageLoad(backbuffer, rotatedTexel );
    vec4 back = imageLoad(backbuffer, texelCoord + ivec2(1, 0));
    //vec4 back = sampleBilinear(backbuffer, rotatedUV);
    //back *= angle < .9 ? 1.: 1. - angle;
    back *= angle < 1. ? 1.: 1. - angle;
    back += vec4(spectrum, 1.);
    imageStore(backbuffer, texelCoord, back);

    finalColor += sampleBilinear(backbuffer, polar.yx).rgb;
    
    value = vec4(finalColor, 1.0);
    value += circle(centered, .01) > 0.? 0.:1.;
	imageStore(imgOutput, texelCoord, value);
}

