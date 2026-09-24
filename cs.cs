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
uniform int nca_channel_view;         // -1 = composite, 0..11 = direct channel w/o CC, 12 = 3 selectable channels (RGB), 13 = 3 main PCA components for RGB
uniform ivec3 custom_channels;         // Selectable 3 channels (r, g, b) in [0..11]
uniform mat3 pca_palette_matrix;       // 3x3 PCA orientation and hue rotation matrix
uniform float pca_pc1[12];            // 1st Principal Component (Red)
uniform float pca_pc2[12];            // 2nd Principal Component (Green)
uniform float pca_pc3[12];            // 3rd Principal Component (Blue)
uniform float pca_mean[12];           // 12-channel mean vector
uniform vec3 pca_std;                 // Standard deviations of top 3 PCs for contrast normalization
uniform float nca_state_dim;          // Direct NCA state damping / dimming factor (0.0 to 1.0)
uniform ivec2 sim_size;               // Simulation resolution (width, height)
uniform int sim_filtering;            // 0 = Smooth Hermite, 1 = Bilinear, 2 = Nearest
uniform int pass_mode;                // 0 = Simulation step, 1 = Screen presentation & continuous resampling

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

// Simulation grid dimensions (initialized per pass in main)
uint SW = 267u;
uint SH = 200u;

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
    int nx = (current_index.x + dx * int(D)) % int(SW);
    if (nx < 0) nx += int(SW);
    int ny = (current_index.y + dy * int(D)) % int(SH);
    if (ny < 0) ny += int(SH);
    return get_xyc(uint(nx), uint(ny), c);
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

vec4 get_cell_color(uint x, uint y) {
    float[12] states_out = get_xy(x, y);
    vec4 col;
    if (nca_channel_view == 12) {
        // Selectable 3 channels mapped to RGB w/o color correction
        int chR = clamp(custom_channels.x, 0, 11);
        int chG = clamp(custom_channels.y, 0, 11);
        int chB = clamp(custom_channels.z, 0, 11);
        col = vec4(clamp(abs(states_out[chR]), 0.0, 1.5),
                   clamp(abs(states_out[chG]), 0.0, 1.5),
                   clamp(abs(states_out[chB]), 0.0, 1.5), 1.0);
    } else if (nca_channel_view == 13) {
        // 3 main PCA components projected to RGB with re-rollable / shiftable palette matrix
        float r_proj = 0.0;
        float g_proj = 0.0;
        float b_proj = 0.0;
        for (int c = 0; c < 12; c++) {
            float centered = states_out[c] - pca_mean[c];
            r_proj += centered * pca_pc1[c];
            g_proj += centered * pca_pc2[c];
            b_proj += centered * pca_pc3[c];
        }
        vec3 p = vec3(
            r_proj / (2.5 * max(0.05, pca_std.x)),
            g_proj / (2.5 * max(0.05, pca_std.y)),
            b_proj / (2.5 * max(0.05, pca_std.z))
        );
        vec3 rotated = pca_palette_matrix * p;
        col = vec4(clamp(rotated + vec3(0.5), 0.0, 1.0), 1.0);
    } else if (nca_channel_view >= 0 && nca_channel_view < 12) {
        // Direct single channel preview w/o color correction
        float chVal = clamp(abs(states_out[nca_channel_view]), 0.0, 1.5);
        col = vec4(chVal, chVal, chVal, 1.0);
    } else {
        // Default composite with color correction
        col = vec4(states_out[3], states_out[0], states_out[3], states_out[3]);
        col.g *= 0.5555;
        col *= 0.9;
    }
    return col;
}

void main() {
    SW = uint(sim_size.x > 0 ? sim_size.x : (imageSize(outputTexture).x / 3));
    SH = uint(sim_size.y > 0 ? sim_size.y : (imageSize(outputTexture).y / 3));
    ivec2 screen_size = imageSize(outputTexture);

    if (pass_mode == 0) {
        // --- PASS 0: NCA SIMULATION STEP (Dispatched over sim_size) ---
        if (gl_GlobalInvocationID.x >= SW || gl_GlobalInvocationID.y >= SH) {
            return;
        }

        current_index = ivec2(gl_GlobalInvocationID.xy);

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

        if (frame == 0) {
            return;
        }

        // Sample backbuffer at coordinate corresponding to current simulation cell
        ivec2 backCoord = ivec2((vec2(current_index) + 0.5) / vec2(float(SW), float(SH)) * vec2(screen_size));
        backCoord = clamp(backCoord, ivec2(0), screen_size - 1);
        vec4 scaled_back = imageLoad(backbuffer, backCoord);

        // Disturb states (scaled by spectrogram_influence; giant imprinted glyphs always disturb)
        float audioDisturb = spectrogram_influence;
        for (uint s = 1u; s < N; s++) {
            float dist = (length(scaled_back) > 3.0) ? (1.0 - length(scaled_back)) * 0.6 : (length(scaled_back) / 5.9 * audioDisturb);
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

        // Direct state dimming / damping directly in simulation buffer
        float dimStrength = 1.0 - clamp(nca_state_dim, 0.0, 1.0);
        if (dimStrength > 0.001) {
            float cooling = 1.0 - dimStrength * 0.035; // gentle dissipation (up to ~3.5% per frame)
            for (uint s = 0u; s < N; s++) {
                state[s] *= cooling;
            }
        }

        set_xy(uint(current_index.x), uint(current_index.y), state);
        return;
    }

    // --- PASS 1: SCREEN PRESENTATION & SMOOTH RESAMPLING (Dispatched over screen_size) ---
    ivec2 fragCoord = ivec2(gl_GlobalInvocationID.xy);
    if (fragCoord.x >= screen_size.x || fragCoord.y >= screen_size.y) {
        return;
    }

    vec2 uv = (vec2(fragCoord) + 0.5) / vec2(screen_size);

    // Resample CA cells using zero-artifact continuous coordinate mapping
    vec2 simCoord = uv * vec2(float(SW), float(SH)) - 0.5;
    vec4 cellColor;

    if (sim_filtering == 2) {
        // Nearest-Neighbor (Clean pixel-art grid w/ toroidal modulo)
        int nx = int(floor(simCoord.x + 0.5));
        int ny = int(floor(simCoord.y + 0.5));
        uint x0 = uint(((nx % int(SW)) + int(SW)) % int(SW));
        uint y0 = uint(((ny % int(SH)) + int(SH)) % int(SH));
        cellColor = get_cell_color(x0, y0);
    } else {
        float fx = fract(simCoord.x);
        float fy = fract(simCoord.y);
        int ix = int(floor(simCoord.x));
        int iy = int(floor(simCoord.y));

        uint x0 = uint(((ix % int(SW)) + int(SW)) % int(SW));
        uint x1 = uint((((ix + 1) % int(SW)) + int(SW)) % int(SW));
        uint y0 = uint(((iy % int(SH)) + int(SH)) % int(SH));
        uint y1 = uint((((iy + 1) % int(SH)) + int(SH)) % int(SH));

        vec4 c00 = get_cell_color(x0, y0);
        vec4 c10 = get_cell_color(x1, y0);
        vec4 c01 = get_cell_color(x0, y1);
        vec4 c11 = get_cell_color(x1, y1);

        float wx, wy;
        if (sim_filtering == 1) {
            // Bilinear
            wx = fx;
            wy = fy;
        } else {
            // Smooth Hermite S-Curve (Default - Zero-artifact, C1 continuous derivative)
            wx = fx * fx * (3.0 - 2.0 * fx);
            wy = fy * fy * (3.0 - 2.0 * fy);
        }

        vec4 top = mix(c00, c10, wx);
        vec4 bot = mix(c01, c11, wx);
        cellColor = mix(top, bot, wy);
    }

    // Audio radar sweep overlay & backbuffer decay
    vec2 centered = (uv - 0.5) * 2.0;
    vec2 ratio = centered * vec2(float(screen_size.x) / float(screen_size.y), 1.0);
    float radius = length(ratio);
    float angle = atan(ratio.y, ratio.x);

    float effectiveSpeed = (scan_speed > 0.0001) ? scan_speed : 0.4;
    float s = (scan_phase >= 0.0) ? scan_phase : fract(time * effectiveSpeed);

    angle = (angle + 3.14159265) / (2.0 * 3.14159265);
    angle = 1.0 - angle;
    angle = angle + 0.75;
    angle = fract(angle);

    float numBins = float(imageSize(audioTexture).x);
    if (numBins < 1.0) numBins = 512.0;
    float fmin = clamp(freq_range.x, 0.0, 1.0);
    float fmax = clamp(freq_range.y, 0.0, 1.0);
    if (fmax <= fmin) fmax = 1.0;

    // Radially map from center (bass) to screen edge (treble)
    // radius = 0 is center (bass fmin), radius ~ 1.05 is screen edge (treble fmax)
    // Eliminates empty center (bass is in the center) and eliminates overly long bins at perimeter
    float radNorm = clamp(radius / 1.05, 0.0, 1.0);
    float freqNorm = mix(fmin, fmax, radNorm);

    // Continuous linear interpolation between FFT bins
    float binPos = freqNorm * (numBins - 1.0);
    int b0 = int(floor(binPos));
    int b1 = min(b0 + 1, int(numBins) - 1);
    float bFrac = binPos - float(b0);
    float mag0 = imageLoad(audioTexture, ivec2(b0, 0)).r;
    float mag1 = imageLoad(audioTexture, ivec2(b1, 0)).r;
    float audioMagnitude = mix(mag0, mag1, bFrac);
    audioMagnitude = clamp(audioMagnitude * 200.0, 0.0, 1.0);

    vec3 color = vec3(audioMagnitude);
    float angleDiff = abs(angle - fract(s));
    angleDiff = min(angleDiff, 1.0 - angleDiff);
    float intensity = (spectrogram_influence > 0.0001 && angleDiff < 0.012) ? 1.0 : 0.0;
    vec3 spectrum = color * intensity * (2.0 * spectrogram_influence);
    vec3 activeAudioSignal = spectrum;

    vec4 back = imageLoad(backbuffer, fragCoord);
    if (spectrogram_influence > 0.0001 && length(activeAudioSignal) > 0.001) {
        float sigAlpha = clamp(length(activeAudioSignal) * 1.5, 0.0, 1.0);
        back = max(back, vec4(activeAudioSignal, sigAlpha));
    }
    float effectiveDecay = (decay_time > 0.0001) ? clamp(decay_time, 0.5, 0.9999) : 0.995;
    back *= effectiveDecay;
    imageStore(backbuffer, fragCoord, back);

    vec3 overlayBack = (spectrogram_influence > 0.0001) ? (back.rgb * 0.35) : (length(back.rgb) > 1.5 ? (back.rgb * 0.35) : vec3(0.0));
    cellColor.rgb += activeAudioSignal * 0.85 + overlayBack;

    imageStore(outputTexture, fragCoord, cellColor);
}
 
