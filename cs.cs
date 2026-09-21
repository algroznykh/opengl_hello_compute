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
uniform float nca_perturb;
uniform float nca_seed;
uniform float nca_step_scale;
uniform float waveform_mix;

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

// Constants
const uint N = 12u;
const float S = 5000.0;
const int[12] B = int[12](-332,-212,-353,566,-86,837,510,-339,373,414,-787,-712);

// Weight matrix W (48x12)
const int[48][12] W = int[48][12](
    int[12](-816,142,-40,142,-119,0,758,70,99,-149,-165,-141),
    int[12](-24,-1182,-67,438,267,190,358,113,191,-64,-196,-263),
    int[12](10,170,-785,-49,98,-39,733,-75,210,-161,-112,-107),
    int[12](60,-183,158,-966,352,-336,588,-356,123,-531,-364,-201),
    int[12](-11,-89,-77,-259,-2788,276,46,56,72,-28,-179,66),
    int[12](24,-43,-1,463,-452,-2262,95,59,-51,3,-222,-216),
    int[12](-162,158,-113,19,-26,-66,-1856,49,-27,-302,-99,-366),
    int[12](58,9,203,-20,-200,-76,-28,-2600,-206,29,127,358),
    int[12](-68,-222,-143,-254,-287,157,255,400,-3107,-87,373,-420),
    int[12](-59,-75,1,-464,19,82,-26,181,-187,-1802,124,-557),
    int[12](191,166,218,-398,22,131,136,552,301,-273,-1490,-123),
    int[12](188,142,283,-677,143,84,-154,-80,212,-34,-69,-1619),
    int[12](257,-95,-4,62,-3,36,29,-17,16,12,34,6),
    int[12](-79,218,-137,-68,-23,-3,63,-20,-12,-14,-1,16),
    int[12](-97,-42,211,91,-3,26,42,12,-13,-40,-5,37),
    int[12](-106,-5,-160,673,-500,177,219,41,-167,-479,-273,-478),
    int[12](-137,-144,-138,-30,402,-100,20,59,-57,-11,-16,93),
    int[12](144,174,145,84,14,128,-180,-234,-127,-221,59,192),
    int[12](8,4,6,76,-92,-343,1,-6,97,-20,-50,-41),
    int[12](-38,-38,-24,42,158,-184,-28,-130,70,-48,28,-43),
    int[12](-178,-190,-164,-131,116,-15,-154,-93,229,57,33,193),
    int[12](-52,-84,-60,-126,86,-50,32,210,-71,268,123,-203),
    int[12](213,184,191,-65,162,-25,128,445,80,69,432,-191),
    int[12](-124,-117,-125,-59,-130,-84,-42,173,261,14,5,-278),
    int[12](51,-269,-169,-412,300,-62,148,-72,219,-66,335,492),
    int[12](-157,83,-128,-397,144,-98,369,17,183,-123,392,490),
    int[12](-193,-370,35,-306,228,-5,155,49,169,-15,384,480),
    int[12](198,172,203,-172,179,92,127,183,450,-2,-66,222),
    int[12](48,39,62,169,1160,-98,174,32,-15,100,117,65),
    int[12](-36,-63,-54,279,-132,-1232,-7,130,90,37,-20,-38),
    int[12](14,54,46,-129,34,85,-480,100,-83,-325,-78,107),
    int[12](-228,-276,-217,-255,-117,-148,0,1,-156,257,164,-8),
    int[12](-27,-95,-41,-34,-35,33,150,269,-1215,-116,299,-217),
    int[12](-294,-409,-300,-267,104,-40,203,101,133,-150,368,8),
    int[12](139,154,113,27,19,274,103,150,217,174,-293,300),
    int[12](82,39,56,-211,-121,-44,-61,557,-982,-167,358,-339),
    int[12](-32,29,20,-6,-28,-19,21,1,0,21,-23,-2),
    int[12](13,-12,38,28,-47,32,-42,38,-37,-14,-27,-65),
    int[12](43,5,-36,-21,10,-21,9,15,-34,-19,-16,-30),
    int[12](528,533,528,-52,-199,-120,-6,140,211,-186,88,29),
    int[12](25,9,27,-71,30,-8,-6,73,31,44,213,-1),
    int[12](227,224,226,70,71,-43,101,100,168,-405,69,-83),
    int[12](-142,-184,-148,-169,-115,119,25,26,266,296,296,218),
    int[12](118,164,96,279,-120,98,41,30,233,-190,-124,204),
    int[12](-73,-95,-63,-89,65,-13,-5,-50,77,-51,38,87),
    int[12](-55,-71,-42,-44,29,6,38,-174,77,5,-45,112),
    int[12](148,73,136,-81,-113,12,226,133,-417,-305,226,-81),
    int[12](100,152,115,28,-229,26,-272,-200,528,-81,-102,235)
);

// Hypergraph tree nodes (origin at (0, 0) in center of circular spectrogram)
const vec2[24] TREE_NODES = vec2[24](
    vec2( 0.00,  0.00), // 0: Root origin at center of spectrogram
    vec2( 0.02,  0.13), // 1: Main trunk
    vec2(-0.11,  0.27), // 2: Left bough fork
    vec2( 0.12,  0.28), // 3: Right bough fork (dominant)
    vec2(-0.25,  0.36), // 4: Left outer limb
    vec2(-0.06,  0.43), // 5: Left inner limb
    vec2( 0.08,  0.47), // 6: Right inner limb
    vec2( 0.27,  0.41), // 7: Right outer limb
    vec2(-0.38,  0.44), // 8: Left outer branch tip
    vec2(-0.31,  0.56), // 9: Left-center secondary twig
    vec2(-0.17,  0.62), // 10: Center-left canopy twig
    vec2(-0.03,  0.59), // 11: Central crest fork
    vec2( 0.06,  0.69), // 12: High central twig
    vec2( 0.21,  0.64), // 13: Upper-right canopy twig
    vec2( 0.38,  0.55), // 14: Right-center secondary twig
    vec2( 0.46,  0.39), // 15: Right outer branch tip
    vec2(-0.26,  0.72), // 16: Left high canopy leaf
    vec2( 0.13,  0.78), // 17: Summit apex leaf
    vec2( 0.33,  0.72), // 18: Right high canopy leaf
    vec2(-0.10, -0.08), // 19: Left basal root fork
    vec2( 0.10, -0.07), // 20: Right basal root fork
    vec2(-0.18, -0.15), // 21: Left root tip
    vec2( 0.18, -0.14), // 22: Right root tip
    vec2( 0.00, -0.06)  // 23: Taproot center
);

// 28 Edges: 23 Tree branches + 5 Hypergraph cross-chords
const ivec2[28] TREE_EDGES = ivec2[28](
    // Trunk & basal roots (originating at center 0, 0)
    ivec2(0, 1), ivec2(0, 19), ivec2(19, 21), ivec2(0, 20), ivec2(20, 22), ivec2(0, 23),
    // Primary boughs
    ivec2(1, 2), ivec2(1, 3),
    // Secondary limbs
    ivec2(2, 4), ivec2(2, 5),
    ivec2(3, 6), ivec2(3, 7),
    // Tertiary branches
    ivec2(4, 8), ivec2(4, 9),
    ivec2(5, 10), ivec2(5, 11),
    ivec2(6, 12), ivec2(6, 13),
    ivec2(7, 14), ivec2(7, 15),
    // Canopy apex tips
    ivec2(9, 16), ivec2(12, 17), ivec2(13, 18),
    // Hypergraph cross-chords (connecting branch clusters across space)
    ivec2(8, 10), ivec2(10, 12), ivec2(12, 14), ivec2(13, 15), ivec2(4, 7)
);

uint D=1; // dilation

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


float sobx(uint c) {
    // return R(-1, 1, c) + R(-1, 0, c)*2.0 + R(-1,-1, c)
    //       -R( 1, 1, c) - R( 1, 0, c)*2.0 - R( 1,-1, c);
    mat3 f = mat3(1, 2, 1, 
                  0, 0, 0, 
                  -1, -2, -1);

    // f = rot(0) * f; 

    float res = f[0][0] * R(-1, 1, c) + f[1][0] * R(-1, 0, c) + f[2][0] * R(-1,-1, c)
              + f[0][1] * R(0, 1, c) + f[1][1] * R(0, 0, c) + f[2][1] * R(0, -1, c)   
              + f[0][2] * R( 1, 1, c) + f[1][2] * R( 1, 0, c) + f[2][2] * R( 1,-1, c);
    return res;
}

float soby(uint c) {
    return R( 1, 1, c) + R( 0, 1, c)*2.0 + R(-1, 1, c)
          -R( 1,-1, c) - R( 0,-1, c)*2.0 - R(-1,-1, c);
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

    // Do 1x1 conv with homeostatic, zero-mean weight perturbation
    float[12] y;
    float effectiveStepScale = clamp((nca_step_scale > 0.0) ? nca_step_scale : 1.0, 0.4, 1.8);
    float perturbAmount = clamp(nca_perturb, 0.0, 1.5);
    uint seedInt = uint(nca_seed * 65536.0);

    for (uint c = 0u; c < N; c++) {
        float val = float(B[c]);

        if (perturbAmount > 0.0001) {
            // Scale perturbation relative to trained weight magnitude to preserve matrix eigenspectrum and signs
            // perturbAmount in [0, 1] gives gentle 0% to 10% relative weight variation
            float relScale = perturbAmount * 0.10;
            float absScale = perturbAmount * 2.5;

            for (uint i = 0u; i < 48u; i++) {
                // Anti-symmetric zero-mean noise pair strictly avoids DC drift across the 48 inputs
                uint h = (i * 73856093u ^ c * 19349663u ^ seedInt);
                h = (h ^ (h >> 13u)) * 1274126177u;
                float rawNoise = float(int(h & 0xFFFFu) - 32768) / 32768.0; // [-1.0, 1.0]
                
                // Anti-symmetric alternating sign guarantees sum of noise across 48 inputs is zero-mean
                float noise = (i % 2u == 0u) ? rawNoise : -rawNoise;
                
                float origW = float(W[i][c]);
                float w = origW * (1.0 + noise * relScale) + noise * absScale;
                val += hs[i] * w;
            }
        } else {
            for (uint i = 0u; i < 48u; i++) {
                val += hs[i] * float(W[i][c]);
            }
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
    float intensity = angleDiff < 0.012 ? 1.0 : 0.0;
    vec3 spectrum = color * intensity * 2.;

    // --- Center-Originating Tree Branch & Hypergraph Edge Waveform ---
    vec3 waveformColor = vec3(0.0);
    float wfMix = clamp(waveform_mix, 0.0, 1.0);
    if (wfMix > 0.0001) {
        // Point p is centered at (0, 0) in the middle of the circular spectrogram
        // Fixed orientation: does NOT rotate!
        vec2 p = centered;
        float rDist = length(p);

        // Render branching tree edges and hypergraph chords (28 edges: 23 tree branches + 5 hypergraph cross-chords)
        for (int i = 0; i < 28; i++) {
            vec2 a = TREE_NODES[TREE_EDGES[i].x];
            vec2 b = TREE_NODES[TREE_EDGES[i].y];
            vec2 ab = b - a;
            float lenSq = dot(ab, ab);
            float t = clamp(dot(p - a, ab) / max(lenSq, 0.0001), 0.0, 1.0);
            vec2 c = a + t * ab;

            float r = length(c);
            float rNorm = clamp(r / 0.82, 0.0, 1.0);

            // Fetch audio waveform sample along this radial distance
            int waveIdx = int(rNorm * (MUSICAL_BINS - 1.0));
            float waveVal = imageLoad(audioTexture, ivec2(waveIdx, 1)).r;
            float waveAmp = clamp(waveVal * 2.8, -1.0, 1.0);

            // Audio waveform acoustic vibration: bends and vibrates edges with sound frequencies
            vec2 n = normalize(vec2(-ab.y, ab.x));
            float tether = sin(t * 3.14159265); // Anchored at nodes t=0 and t=1
            float edgeVib = tether * (waveAmp * 0.025 + 0.006 * sin(t * 14.0 + time * 8.0 + r * 16.0));
            vec2 cVib = c + n * edgeVib;

            float d = length(p - cVib);

            bool isChord = (i >= 23);
            float baseThick = isChord ? 0.0016 : mix(0.0055, 0.0016, rNorm);

            // Crisp antialiased core filament
            float core = smoothstep(baseThick, baseThick * 0.25, d);

            // Bounded soft aura glow (falls cleanly to 0 outside radius)
            float aura = smoothstep(baseThick * 3.6, 0.0, d) * 0.45;

            // Bioluminescent energy pulse propagating outward from center root along the branch
            float pulse = sin(r * 24.0 - time * 8.0 - waveAmp * 3.0);
            pulse = pow(clamp(pulse, 0.0, 1.0), 5.0) * (0.3 + 0.7 * abs(waveAmp));

            float edgeLuma = core * 1.8 + aura + pulse * 1.6 * core;

            // Palette: organic emerald-cyan at root -> electric azure-cyan at canopy tips
            vec3 edgeCol = mix(vec3(0.08, 0.95, 0.65), vec3(0.18, 0.85, 1.0), rNorm);
            if (isChord) {
                edgeCol = mix(edgeCol, vec3(0.85, 0.38, 1.0), 0.65); // Hypergraph chords: luminous violet-magenta
            }
            edgeCol += vec3(0.85, 0.98, 1.0) * pulse * 0.8; // Bright pulse spark

            waveformColor += edgeCol * edgeLuma;
        }

        // Render Hypergraph Vertices / Junction Nodes (24 nodes)
        for (int k = 0; k < 24; k++) {
            vec2 vk = TREE_NODES[k];
            float rk = length(vk);
            int wIdx = int(clamp(rk / 0.82, 0.0, 1.0) * (MUSICAL_BINS - 1.0));
            float wVal = imageLoad(audioTexture, ivec2(wIdx, 1)).r;

            float dNode = length(p - vk);
            float nRadius = (k == 0) ? 0.020 : (k <= 3 ? 0.010 : 0.0065);
            float nCore = smoothstep(nRadius, 0.0, dNode);
            float nGlow = smoothstep(nRadius * 2.8, 0.0, dNode) * 0.55;

            float nodePulse = 1.0 + 2.5 * abs(wVal);
            vec3 nCol = (k == 0) ? vec3(0.3, 1.0, 0.7) : ((k >= 19) ? vec3(0.2, 0.85, 0.6) : vec3(0.55, 0.92, 1.0));
            waveformColor += nCol * (nCore * 2.2 + nGlow) * nodePulse;
        }

        // Central Root Origin Hub (radiant anchor at 0, 0)
        float rootGlow = smoothstep(0.08, 0.0, rDist) * (1.2 + 2.0 * abs(imageLoad(audioTexture, ivec2(0, 1)).r));
        waveformColor += vec3(0.12, 0.92, 0.68) * rootGlow;

        // Mask to circular spectrogram display bounds
        waveformColor *= smoothstep(0.88, 0.82, rDist);
    }

    // Blend between circular frequency spectrogram and center-originating waveform
    vec3 activeAudioSignal = mix(spectrum, waveformColor, wfMix);
    vec3 finalColor;
    
    vec4 back = imageLoad(backbuffer, texelCoord); 
    // Only refresh backbuffer energy where audio signal is non-zero
    float sigAlpha = clamp(length(activeAudioSignal) * 1.5, 0.0, 1.0);
    back = max(back, vec4(activeAudioSignal, sigAlpha));

    // Radar decay persistence control (defaults to 0.995)
    float effectiveDecay = (decay_time > 0.0001) ? clamp(decay_time, 0.5, 0.9999) : 0.995;
    back *= effectiveDecay;
    imageStore(backbuffer, texelCoord, back);

    finalColor += back.rgb;
    // Add direct waveform highlight for crisp, immediate center-originating luminescence
    finalColor += waveformColor * wfMix * 0.75;
    
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
            lap(0u),// + value.r * 2. - tex.x * radius * 5.,
            lap(1u),// + value.g  - tex.y * radius * 5.,
            lap(2u),// + value.b  - tex.z * radius * 5.,
            lap(3u),// - tex.b * 2.,
            soby(4u) * .5 + sobx(4u) * .5,// - tex.r * 2.,
            soby(5u) * .5 + sobx(5u) * .5, 
            soby(6u) * .5 + sobx(6u) * .5, 
            soby(7u) * .5 + sobx(7u) * .5,
            sobx(8u) * .5 + soby(8u)  * .5,// - tex.b * 2.,
            sobx(9u) * .5 + soby(9u)  * .5,
            sobx(10u)* .5 + soby(10u) * .5,
            sobx(11u)* .5 + soby(11u) * .5
        );
        
        // Update state
        float[12] xs = get_xy(uint(current_index.x), uint(current_index.y));    
        float[12] state = update(xs, ps);
        vec4 scaled_back = imageLoad(backbuffer, current_index * int(FACTOR));

        if (frame == 0) {
        return;
        }
       
        // disturb states
        for (uint s = 1u; s < N; s++) {
            state[s] += (length(scaled_back) > 3.) ? (1. - length(scaled_back)) *  .6: length(scaled_back)/5.9 ;
        }

        // Auto-rejuvenate dormant/collapsed cells when radar or waveform energy passes over
        float stateEnergy = abs(state[0]) + abs(state[1]) + abs(state[2]) + abs(state[3]);
        if (stateEnergy < 0.02 && length(scaled_back) > 0.08) {
            state[0] = 0.35 * sin(float(current_index.x) * 0.12);
            state[3] = 0.35 * cos(float(current_index.y) * 0.12);
            state[4] = 0.15;
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
    vec4 xrgb = vec4(states_out[3], states_out[0], states_out[3], states_out[3]);
    

    //xrgb *=  (length(xrgb) * .5 - pow(length(centered), 2.) ); 
    // xrgb *= length(value) > 3.9 ? length(value) : 0.  ;
    //xrgb = vec4(finalColor, 1.);

    //xrgb -= (.8 - length(centered));

    //xrgb *= tex * .;//* pow((1. - length(centered)), 1.) * 12.;

    //xrgb = 1. -  vec4(finalColor, 1.) / 4.;

    //xrgb *= (tex * (1. - radius)) + value ;
    //xrgb *= 1. - radius;

    //xrgb *= max(tex, cameraColor);
    //xrgb += spectrum.xxxx;
    //xrgb.r = fract(angle - time);
    //xrgb = value;


    xrgb.g *= .5555;
    xrgb *= .9;
    //xrgb = pow(xrgb, vec4(3.));
    imageStore(outputTexture, fragCoord, xrgb);
    // interpolation
    vec4 interpolated = interp(fragCoord, int(FACTOR-1));
    // Overlay direct luminous radar sweep / tree waveform on top of NCA substrate
    interpolated.rgb += activeAudioSignal * 0.85 + back.rgb * 0.15;
    imageStore(outputTexture, fragCoord, interpolated);
}
 
