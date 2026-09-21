#version 430 core

layout (local_size_x = 10, local_size_y = 10, local_size_z = 1) in;

// Textures and images - matching compute.cpp glBindImageTexture calls
layout(rgba32f, binding = 0) uniform image2D outputTexture;          // glBindImageTexture(0, ...)
layout(r32f, binding = 1) uniform image2D audioTexture;              // glBindImageTexture(1, ...)  
layout(rgba8, binding = 2) uniform image2D camTexture;               // glBindImageTexture(2, ...)
layout(rgba32f, binding = 3) uniform image2D backbuffer;


layout (location = 0) uniform float time;                 /** Time */
layout (location = 1) uniform int frame;                
uniform float scan_speed;
uniform float decay_time;
uniform vec2 freq_range;
uniform float freq_min_hz;
uniform float freq_max_hz;
uniform int scale_type;
uniform float scan_phase;
uniform float nca_step_scale;
uniform float nca_topology; // 0.0 = Cartesian, 1.0 = Polar (pole in center)
uniform float nca_pole_radial;    // +1.0 = outward pole, -1.0 = reversed/inward pole
uniform float nca_pole_azimuthal; // +1.0 = standard CCW spin, -1.0 = reversed CW spin
uniform float spectrogram_influence; // Controls how much spectrogram affects NCA state & beam visibility (0.0 to 2.0)
uniform int nca_channel_view;         // -1 = composite (normal), 0..11 = direct channel w/o color correction, 12 = raw RGB

// Uniforms buffer - binding 4 (avoiding conflict with image bindings)
layout(std140, binding = 4) uniform Uniforms {
    mat3 kernel;
    uint filter_type;
    uint uframe;
    uint agentCount;
};


// Storage buffer for states - binding 5 (avoiding conflict with image bindings)
layout(std430, binding = 5) restrict buffer StateBuffer {
    float states[][12];
};

// Agent simulation buffers - binding 6-10
layout(std430, binding = 6) restrict buffer PositionBuffer {
    vec2 positions[];
};

layout(std430, binding = 7) restrict buffer VelocityBuffer {
    vec2 velocities[];
};

layout(std430, binding = 8) restrict buffer TrailGridBuffer {
    float trailGrid[];
};

layout(std430, binding = 9) restrict buffer AgentGridBuffer {
    vec4 agentGrid[];
};

layout(std430, binding = 10) restrict buffer AngleBuffer {
    float angles[];
};

// Weight buffer for Human-Assisted Genetic Algorithm - binding 11
layout(std430, binding = 11) restrict buffer NcaWeightBuffer {
    float W_weights[576];
    float B_biases[12];
};

// Constants
const uint N = 12u;
const float S = 5000.0;
uint D = 1; // dilation

// Simulation size
uint FACTOR = 3;
uint SW = imageSize(outputTexture).x / FACTOR;
uint SH = imageSize(outputTexture).y / FACTOR;

// Global variable for current index
ivec2 current_index;

// Helper functions

ivec2 toPolar(vec2 fragCoord, vec2 center) {
    // Vector from center to fragment
    vec2 diff = fragCoord - center;

    // Calculate radius (distance from center)
    float radius = length(diff);

    // Calculate angle in radians, then convert to degrees
    float angleRad = atan(diff.y, diff.x);
    float angleDeg = degrees(angleRad);

    // Normalize angle to [0, 360) range
    if (angleDeg < 0.0) {
        angleDeg += 360.0;
    }

    return ivec2(int(radius), int(angleDeg));
}

float smin( float a, float b, float k )
{
    k *= 2.0;
    float x = b-a;
    return 0.5*( a+b-sqrt(x*x+k*k) );
}

float circle(vec2 p, float r) {
    return length(p) - r;
}

float sdPolarCircle(float r_pos, float theta_pos, float radius) {
    float d_sq = r_pos*r_pos - 2.0*r_pos*cos(- theta_pos);
    return sqrt(d_sq) - radius;
}

float[12] get_xy(uint x, uint y) {
    uint i = x + y * SW;
    return states[i];
}

float get_xyc(uint x, uint y, uint c) {
    uint i = x + y * SW;
    return states[i][c];
}

void set_xy(uint x, uint y, float[12] cs) {
    uint i = x + y * SW;
    states[i] = cs;
}

float R(int dx, int dy, uint c) {
    uint x = (uint(current_index.x + dx*D) + SW) % SW;
    uint y = (uint(current_index.y + dy*D) + SH) % SH;
    return get_xyc(x, y, c);
}


vec2 rotate(vec2 vec, float angle) {
    float cs = cos(angle);
    float sn = sin(angle);
    return vec2(
        vec.x * cs - vec.y * sn,
        vec.x * sn + vec.y * cs
    );
}

float triangle(vec2 pin, float r) {
    float k = sqrt(3.0);
    vec2 p = pin;
    p.x = abs(p.x) - r;
    p.y = p.y + r/k;
    if(p.x + k*p.y > 0.0) {
        p = vec2(p.x - k*p.y, -k*p.x - p.y) / 2.0;
    }
    p.x -= clamp(p.x, -2.0*r, 0.0);
    return -length(p) * sign(p.y);
}

float box(vec2 p, vec2 b) {
    vec2 d = abs(p) - b;
    return length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0);
}

mat2 rot(float q) {
    return mat2(cos(q), -sin(q), sin(q), cos(q));
}

float lap(uint c) {
    return R(1,1,c) + R(1,-1,c) + R(-1,1,c) + R(-1,-1,c) 
        + 2.0 * (R(0,1,c) + R(0,-1,c) + R(1,0,c) + R(-1,0,c)) - 12.0*R(0,0,c);
}


float raw_sobx(uint c) {
    return (R(-1, 1, c) + R(-1, 0, c)*2.0 + R(-1,-1, c))
         - (R( 1, 1, c) + R( 1, 0, c)*2.0 + R( 1,-1, c));
}

float raw_soby(uint c) {
    return (R( 1, 1, c) + R( 0, 1, c)*2.0 + R(-1, 1, c))
         - (R( 1,-1, c) + R( 0,-1, c)*2.0 + R(-1,-1, c));
}

// Polar rotated Sobel filters w.r.t. coordinate (pole in center)
float sobx(uint c) {
    if (abs(nca_topology) < 0.001) {
        return raw_sobx(c);
    }
    vec2 caCenter = vec2(float(SW), float(SH)) * 0.5;
    vec2 caDiff = vec2(current_index) - caCenter;
    float caAngle = atan(caDiff.y, caDiff.x);
    float phi = caAngle * nca_topology;
    float effRadialSign = (abs(nca_pole_radial) > 0.01) ? sign(nca_pole_radial) : 1.0;
    float radial = (cos(phi) * raw_sobx(c) + sin(phi) * raw_soby(c)) * effRadialSign;
    return mix(raw_sobx(c), radial, abs(nca_topology));
}

float soby(uint c) {
    if (abs(nca_topology) < 0.001) {
        return raw_soby(c);
    }
    vec2 caCenter = vec2(float(SW), float(SH)) * 0.5;
    vec2 caDiff = vec2(current_index) - caCenter;
    float caAngle = atan(caDiff.y, caDiff.x);
    float phi = caAngle * nca_topology;
    float effAzimSign = (abs(nca_pole_azimuthal) > 0.01) ? sign(nca_pole_azimuthal) : 1.0;
    float azimuthal = (-sin(phi) * raw_sobx(c) + cos(phi) * raw_soby(c)) * effAzimSign;
    return mix(raw_soby(c), azimuthal, abs(nca_topology));
}

float[12] update(float[12] xs, float[12] ps) {
    // Construct hidden state
    float[48] hs;
    for (uint i = 0u; i < N; i++) {
        hs[i] = xs[i];
        hs[i+N] = ps[i];
        hs[i+N*2u] = abs(xs[i]);
        hs[i+N*3u] = abs(ps[i]);
    }

    // Do 1x1 conv with genetic algorithm dynamic weights from SSBO binding 11
    float[12] y;
    float effectiveStepScale = clamp((nca_step_scale > 0.0) ? nca_step_scale : 1.0, 0.4, 1.8);

    for (uint c = 0u; c < N; c++) {
        float val = B_biases[c];
        for (uint i = 0u; i < 48u; i++) {
            val += hs[i] * W_weights[i * 12u + c];
        }
        
        y[c] = xs[c] + (val / S) * effectiveStepScale;

        // Soft-limiter near boundaries prevents hard saturation and dead states
        if (abs(y[c]) > 1.35) {
            y[c] *= 0.98;
        }
        y[c] = clamp(y[c], -1.5, 1.5);
    }

    // Prevent channel 4 from getting permanently hard-locked to zero
    if (abs(y[4]) < 0.001) {
        y[4] = 0.002;
    }

    return y;
}

vec4 camlap(ivec2 coord) {
    mat3 lapmat = mat3(1, 2, 1, 2, -12, 2, 1, 2, 1);
    vec4 res = vec4(0.0);
    for (int i = -1; i < 2; i++) {
        for (int j = -1; j < 2; j++) {
            ivec2 sampleCoord = coord + ivec2(i, j);
            sampleCoord = clamp(sampleCoord, ivec2(0), imageSize(camTexture) - 1);
            res += imageLoad(camTexture, sampleCoord) * lapmat[i+1][j+1];
        }
    }
    return res;
}

vec4 bilinearInterp(vec2 coord) {
    vec2 f = fract(coord);
    ivec2 i = ivec2(floor(coord));
    
    // Clamp coordinates to texture bounds
    ivec2 size = imageSize(outputTexture) - 1;
    
    vec4 tl = imageLoad(outputTexture, clamp(i, ivec2(0), size));
    vec4 tr = imageLoad(outputTexture, clamp(i + ivec2(1, 0), ivec2(0), size));
    vec4 bl = imageLoad(outputTexture, clamp(i + ivec2(0, 1), ivec2(0), size));
    vec4 br = imageLoad(outputTexture, clamp(i + ivec2(1, 1), ivec2(0), size));
    
    vec4 top = mix(tl, tr, f.x);
    vec4 bottom = mix(bl, br, f.x);
    
    return mix(top, bottom, f.y);
}

vec4 interp(ivec2 coord, int n) {
    vec4 res = vec4(0.0);
    for (int i = -n; i < n+1; i++) {
        for (int j = -n; j < n+1; j++) {
            ivec2 sampleCoord = coord + ivec2(i, j);
            sampleCoord = clamp(sampleCoord, ivec2(0), imageSize(outputTexture) - 1);
            res += imageLoad(outputTexture, sampleCoord) ;
        }
    }
    return res/pow(n+2, 2.);
}

/* vec4 sampleBilinear(image2D img, vec2 uv) {
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

vec4 sampleLoG(image2D img, vec2 uv) {
    vec2 texSize = vec2(gl_NumWorkGroups.xy * gl_WorkGroupSize.xy);
    vec2 pixelCoord = uv * texSize;
    ivec2 coord = ivec2(pixelCoord);
    
    // 5x5 Laplacian of Gaussian kernel (approximation)
    // Values normalized for unity gain
    float kernel[25] = float[25](
        0.0,  0.0, -1.0,  0.0,  0.0,
        0.0, -1.0, -2.0, -1.0,  0.0,
       -1.0, -2.0, 16.0, -2.0, -1.0,
        0.0, -1.0, -2.0, -1.0,  0.0,
        0.0,  0.0, -1.0,  0.0,  0.0
    );
    
    vec4 result = vec4(0.0);
    
    for (int y = -2; y <= 2; y++) {
        for (int x = -2; x <= 2; x++) {
            ivec2 sampleCoord = coord + ivec2(x, y);
            sampleCoord = clamp(sampleCoord, ivec2(0), ivec2(texSize) - 1);
            
            int kernelIndex = (y + 2) * 5 + (x + 2);
            result += imageLoad(img, sampleCoord) * kernel[kernelIndex];
        }
    }
    
    return result / 16.0; // Normalize
} */

void main() {
    ivec2 screen_size = imageSize(outputTexture);
    ivec2 fragCoord = ivec2(gl_GlobalInvocationID.xy);
	ivec2 texelCoord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 imgSize = screen_size;
    vec2 uv = (vec2(texelCoord.xy) + 0.5) / vec2(screen_size);
    vec2 centered = (uv - 0.5) * 2.0;
    vec2 ratio = centered * vec2(float(screen_size.x) / float(screen_size.y), 1.0);
    centered = ratio;

    float radius = length(centered);
    float angle = atan(centered.y, centered.x);
    
    // Scanning phase: continuously integrated on CPU so it never jumps backwards when speed changes
    float effectiveSpeed = (scan_speed > 0.0001) ? scan_speed : 0.4;
    float s = (scan_phase >= 0.0) ? scan_phase : fract(time * effectiveSpeed);
    
    // Normalize angle to [0, 1] range
    angle = (angle + 3.14159) / (2.0 * 3.14159);
    angle = 1. - angle;
    angle = angle + .75;
    angle = fract(angle); // loop at midnight

    vec2 polar = vec2(radius, angle) ;

    // Sample camera input
    vec4 cameraColor = imageLoad(camTexture, texelCoord);

    // Get audio data - map angle/radius to musical scale bins for circular spectrogram
    float MUSICAL_BINS = 88.0 * 4.; // Fixed number of musical scale bins

    // Frequency range control: map radial distance across selected normalized frequency window [fmin, fmax]
    float fmin = clamp(freq_range.x, 0.0, 1.0);
    float fmax = clamp(freq_range.y, 0.0, 1.0);
    if (fmax <= fmin) fmax = 1.0;
    float radNorm = clamp((1.0 - radius / 0.8), 0.0, 1.0);
    float freqNorm = mix(fmin, fmax, radNorm);
    int audioIndex = int(freqNorm * (MUSICAL_BINS - 1.0));
    audioIndex = clamp(audioIndex, 0, int(MUSICAL_BINS)-1);
    
    int SPECTRUM_SHIFT = 0;
    //audioIndex = int((1. - radius) * SH) * 8.;
    float audioMagnitude = imageLoad(audioTexture, ivec2(audioIndex + SPECTRUM_SHIFT, 0)).r;
    audioMagnitude = clamp(audioMagnitude * 200.0, 0.0, 1.0); // Amplify and clamp
    
    vec3 color = vec3(audioMagnitude);
    // Shortest angular distance to beam on circle [0, 1] (eliminates glitch at 0/1 seam)
    float angleDiff = abs(angle - fract(s));
    angleDiff = min(angleDiff, 1.0 - angleDiff);
    // When spectrogram_influence is 0, the scanning line is completely hidden
    float intensity = (spectrogram_influence > 0.0001 && angleDiff < 0.012) ? 1.0 : 0.0;
    vec3 spectrum = color * intensity * (2.0 * spectrogram_influence);

    // Active audio signal is the circular radar spectrogram
    vec3 activeAudioSignal = spectrum;
    vec3 finalColor;
    
    vec4 back = imageLoad(backbuffer, texelCoord); 
    // Only refresh backbuffer energy where audio signal is non-zero
    if (spectrogram_influence > 0.0001 && length(activeAudioSignal) > 0.001) {
        float sigAlpha = clamp(length(activeAudioSignal) * 1.5, 0.0, 1.0);
        back = max(back, vec4(activeAudioSignal, sigAlpha));
    }

    // Radar decay persistence control (defaults to 0.995)
    float effectiveDecay = (decay_time > 0.0001) ? clamp(decay_time, 0.5, 0.9999) : 0.995;
    back *= effectiveDecay;
    imageStore(backbuffer, texelCoord, back);

    finalColor += back.rgb;
    
    vec4 value = vec4(finalColor, 1.0);
    value *= 1. - radius;

    vec4 tex = camlap(fragCoord);
    if (gl_GlobalInvocationID.x >= uint(screen_size.x) || gl_GlobalInvocationID.y >= uint(screen_size.y)) { 
        return; 
    }

    
    //float dial = step(circle(ratio, .75), .0);
    float dial =  .05 - circle(ratio, .85);
    dial = dial > 0. ? dial : 0.;
    int nc = 16;
    float shift = .2;
    float sr = .12;
    float ccc;
    for (int i=1; i<=nc; i++) {
        //float cc = step(circle(ratio + .9 * vec2(sin(shift + i/float(nc) * 2.*acos(-1.)), cos(shift + i/float(nc) * 2*acos(-1.))), sr), .0);

        float cc = - circle(ratio + .9 * vec2(sin(shift + i/float(nc) * 2.*acos(-1.)), cos(shift + i/float(nc) * 2*acos(-1.))), sr);
        //cc += cc * spectrum.x * 100.;
        //dial += cc;
        ccc += cc > 0. ? cc * 4. : 0. ;
        //dial = step(dial, .1);
    }
    //dial = smin(dial, ccc, 0.2);
    dial = dial;// + ccc;
    //dial = max(dial,ccc);
    //dial *= 10.;
    dial = smoothstep(dial, -.05, .01);
    //dial = smoothstep(dial, -0.9, -.92);

    if (gl_GlobalInvocationID.x < SW  && gl_GlobalInvocationID.y < SH ) { 
        current_index = ivec2(int(gl_GlobalInvocationID.x), int(gl_GlobalInvocationID.y));

        float[12] ps = float[12](
            lap(0u),
            lap(1u),
            lap(2u),
            lap(3u),
            sobx(4u),
            sobx(5u), 
            sobx(6u), 
            sobx(7u),
            soby(8u),
            soby(9u),
            soby(10u),
            soby(11u)
        );
        
        // Update state
        float[12] xs = get_xy(uint(current_index.x), uint(current_index.y));    
        float[12] state = update(xs, ps);
        vec4 scaled_back = imageLoad(backbuffer, current_index * int(FACTOR));

        if (frame == 0) {
        return;
        }
       
        // disturb states (scaled by spectrogram_influence; giant imprinted glyphs always disturb)
        float audioDisturb = spectrogram_influence;
        for (uint s = 1u; s < N; s++) {
            float dist = (length(scaled_back) > 3.) ? (1. - length(scaled_back)) * .6 : (length(scaled_back) / 5.9 * audioDisturb);
            state[s] += dist;
        }

        // Auto-rejuvenate dormant/collapsed cells when radar energy or imprint passes over
        float stateEnergy = abs(state[0]) + abs(state[1]) + abs(state[2]) + abs(state[3]);
        if (stateEnergy < 0.02) {
            if (length(scaled_back) > 3.0 || (audioDisturb > 0.001 && length(scaled_back) > 0.08)) {
                state[0] = 0.35 * sin(float(current_index.x) * 0.12);
                state[3] = 0.35 * cos(float(current_index.y) * 0.12);
                state[4] = 0.15;
            }
        }

        set_xy(uint(current_index.x), uint(current_index.y), state);
    }

    // Rescale buffer 
    uint idxs = uint(float(gl_GlobalInvocationID.x) / float(screen_size.x) * float(SW));
    uint idys = uint(float(gl_GlobalInvocationID.y) / float(screen_size.y) * float(SH));
    //idxs = gl_GlobalInvocationID.x;
    //idys = gl_GlobalInvocationID.y;

    // Output to screen
    float[12] states_out = get_xy(idxs, idys);
    vec4 xrgb;
    if (nca_channel_view >= 0 && nca_channel_view < 12) {
        // Direct channel view w/o color correction (raw activation magnitude in grayscale)
        float chVal = clamp(abs(states_out[nca_channel_view]), 0.0, 1.5);
        xrgb = vec4(chVal, chVal, chVal, 1.0);
    } else if (nca_channel_view == 12) {
        // Direct raw RGB channels (0, 1, 2) w/o color correction
        xrgb = vec4(clamp(abs(states_out[0]), 0.0, 1.5),
                    clamp(abs(states_out[1]), 0.0, 1.5),
                    clamp(abs(states_out[2]), 0.0, 1.5), 1.0);
    } else {
        // Default composite with color correction
        xrgb = vec4(states_out[3], states_out[0], states_out[3], states_out[3]);
        xrgb.g *= .5555;
        xrgb *= .9;
    }
    //xrgb = pow(xrgb, vec4(3.));
    imageStore(outputTexture, fragCoord, xrgb);
    // interpolation
    vec4 interpolated = interp(fragCoord, int(FACTOR-1));
    // Overlay direct luminous radar sweep and glowing backbuffer structures (imprinted messages)
    vec3 overlayBack = (spectrogram_influence > 0.0001) ? (back.rgb * 0.35) : (length(back.rgb) > 1.5 ? (back.rgb * 0.35) : vec3(0.0));
    interpolated.rgb += activeAudioSignal * 0.85 + overlayBack;
    imageStore(outputTexture, fragCoord, interpolated);
}
 
