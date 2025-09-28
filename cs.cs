#version 430 core

layout (local_size_x = 10, local_size_y = 10, local_size_z = 1) in;

// Textures and images - matching compute.cpp glBindImageTexture calls
layout(rgba32f, binding = 0) uniform image2D outputTexture;          // glBindImageTexture(0, ...)
layout(r32f, binding = 1) uniform image2D audioTexture;              // glBindImageTexture(1, ...)  
layout(rgba8, binding = 2) uniform image2D camTexture;               // glBindImageTexture(2, ...)
layout(rgba32f, binding = 3) uniform image2D backbuffer;


layout (location = 0) uniform float time;                 /** Time */
layout (location = 1) uniform int frame;

// Uniforms buffer - binding 4 (avoiding conflict with image bindings)
layout(std140, binding = 4) uniform Uniforms {
    mat3 kernel;
    uint filter_type;
    //uint frame;
    //uint agentCount;
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

// Simulation size

// Helper functions

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


// Agent simulation functions
float r(float n) {
    float x = sin(n) * 43758.5453;
    return fract(x);
}

int gridIndex(vec2 p) {
    vec2 pos = p;
    return int(pos.x) + int(pos.y);
}

vec2 rotate(vec2 vec, float angle) {
    float cs = cos(angle);
    float sn = sin(angle);
    return vec2(
        vec.x * cs - vec.y * sn,
        vec.x * sn + vec.y * cs
    );
}

// Torus wrapping function
vec2 torusWrap(vec2 pos, float size) {
    return mod(pos + size, size);
}

// Torus-aware index calculation
int torusIndex(vec2 pos, float size) {
    vec2 wrapped = torusWrap(pos, size);
    return int(wrapped.x) + int(wrapped.y) * int(size);
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
}

void main() {
    ivec2 screen_size = imageSize(outputTexture);
    ivec2 fragCoord = ivec2(gl_GlobalInvocationID.xy);
	ivec2 texelCoord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 imgSize = screen_size;
    vec2 uv = vec2(texelCoord.xy) / vec2(imgSize);
    vec2 centered = uv - 0.5;
    centered *= 2.;
    vec2 ratio = centered * vec2(float(screen_size.x)/float(screen_size.y), 1.);
    centered = ratio;

    ivec2 texSize = imageSize(outputTexture);


    float W = float(screen_size.x);
    float H = float(screen_size.y);
    
    // Agent simulation - each thread handles one agent in parallel
    uint agentId = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x;
    
    uint agentCount = 30000000;
    // Process agents in parallel
    if (agentId < agentCount) {
        vec2 p = positions[agentId];
        vec2 v = velocities[agentId];
        
        // Initialize/reset agents - force initialization for first 100 agents
        if (length(p) < 1.0 || frame < 1) {  // Extended initialization period
            float seed = float(agentId) / agentCount * 10000;  // Fixed denominator
            vec2 center = vec2(W / 2.0, H / 2.0);
            float angle = 3.1415 * 2.0 * r(seed);
            p = center + min(W, H)/166.0 * vec2(cos(angle), sin(angle));  // Closer to center
            //v = normalize(vec2(cos(angle), sin(angle))) * 1.0;    // Simpler velocity
            v = normalize(p - center) * 600. * r(p.x);    
            positions[agentId] = p;
            velocities[agentId] = v;
            angles[agentId] = angle;
        } else {
            // Move agent with proper torus wrapping
            p += v;
            p.x = mod(p.x + W, W);  // Handle negative values properly
            p.y = mod(p.y + H, H);  // Handle negative values properly
            positions[agentId] = p;

            // Deposit trail using torus-aware indexing
            int pIndex = int(p.x) + int(p.y) * int(W);
            if (pIndex >= 0 && pIndex < trailGrid.length()) {
                trailGrid[pIndex] += 5.0; // Increased for visibility
            }

            // Sensor parameters from WGSL
            float x = 1.0 + 9.0 * (p.x / W);
            float sa = acos(-1.0) *  box( 1 * sin((p/vec2(W, H) - 0.5) * 9.0), vec2(0.6, 0.4)) / 6.0;
            float y = 1.0 + 9.0 * (p.y / H);
            float so = 2.0 * acos(-1.0) * sin(box(sin((p/vec2(W, H) - 0.5) * 9.0), vec2(0.6, 0.4)) + 0.1);

            // Front sensor with torus wrapping
            vec2 fP = p + normalize(v) * so;
            int fIndex = int(mod(fP.x + W, W)) + int(mod(fP.y + H, H)) * int(W);
            float f = 0.0;
            if (fIndex >= 0 && fIndex < trailGrid.length()) {
                f = trailGrid[fIndex];
                agentGrid[fIndex] = vec4(0.0, 1.0, 0.0, 1.0);
            }

            // Left sensor with torus wrapping
            vec2 flP = p + rotate(normalize(v) * so, -sa);
            int flIndex = int(mod(flP.x + W, W)) + int(mod(flP.y + H, H)) * int(W);
            float fl = 0.0;
            if (flIndex >= 0 && flIndex < trailGrid.length()) {
                fl = trailGrid[flIndex];
                agentGrid[flIndex] = vec4(1.0, 1.0, 0.0, 1.0);
            }

            // Right sensor with torus wrapping
            vec2 frP = p + rotate(normalize(v) * so, sa);
            int frIndex = int(mod(frP.x + W, W)) + int(mod(frP.y + H, H)) * int(W);
            float fr = 0.0;
            if (frIndex >= 0 && frIndex < trailGrid.length()) {
                fr = trailGrid[frIndex];
                agentGrid[frIndex] = vec4(0.0, 1.0, 1.0, 1.0);
            }

            // Steering logic from WGSL
            if(f > fl && f > fr) {
                // keep going forward
            } else if(f < fl && f < fr) {
                float seed = float(agentId) / float(agentCount) + time;
                float angle = sa * (2.0 * round(r(seed)) - 1.0);
                angles[agentId] += angle;
                v = rotate(v, angle);
            } else if(fl < fr) {
                v = rotate(v, sa);
                angles[agentId] += sa;
            } else if(fr < fl) {
                v = rotate(v, -sa);
                angles[agentId] -= sa;
            }

            if (length(v) > 2.0) {
                v = -normalize(v);
            }
            velocities[agentId] = v;
        }
    }

    // Synchronize all agent threads
    barrier();
    memoryBarrierBuffer();

    // Diffusion pass - each thread handles one pixel with torus topology
    vec2 pixelPos = vec2(gl_GlobalInvocationID.xy);
    if (pixelPos.x < W && pixelPos.y < H) {
        
        int i = int(pixelPos.x) + int(pixelPos.y) * int(W);
        if (i >= 0 && i < trailGrid.length()) {
            float current = trailGrid[i];
            float sum = 0.0;
            
            // 3x3 convolution kernel with torus wrapping
            sum += trailGrid[int(mod(pixelPos.x - 1.0 + W, W)) + int(mod(pixelPos.y - 1.0 + H, H)) * int(W)] * 0.05;
            sum += trailGrid[int(mod(pixelPos.x + W, W)) + int(mod(pixelPos.y - 1.0 + H, H)) * int(W)] * 0.1;
            sum += trailGrid[int(mod(pixelPos.x + 1.0 + W, W)) + int(mod(pixelPos.y - 1.0 + H, H)) * int(W)] * 0.05;
            sum += trailGrid[int(mod(pixelPos.x - 1.0 + W, W)) + int(mod(pixelPos.y + H, H)) * int(W)] * 0.1;
            sum += current * 0.4;
            sum += trailGrid[int(mod(pixelPos.x + 1.0 + W, W)) + int(mod(pixelPos.y + H, H)) * int(W)] * 0.1;
            sum += trailGrid[int(mod(pixelPos.x - 1.0 + W, W)) + int(mod(pixelPos.y + 1.0 + H, H)) * int(W)] * 0.05;
            sum += trailGrid[int(mod(pixelPos.x + W, W)) + int(mod(pixelPos.y + 1.0 + H, H)) * int(W)] * 0.1;
            sum += trailGrid[int(mod(pixelPos.x + 1.0 + W, W)) + int(mod(pixelPos.y + 1.0 + H, H)) * int(W)] * 0.05;
            
            // Store diffused result in agentGrid.x as temporary buffer
            agentGrid[i].x = sum * 0.9;  // Add slight decay
        }
    }
    
    // Synchronize all diffusion writes
    barrier();
    memoryBarrierBuffer();
    
    // Copy back from temporary buffer to trailGrid
    if (pixelPos.x < W && pixelPos.y < H) {
        int i = int(pixelPos.x) + int(pixelPos.y) * int(W);
        if (i >= 0 && i < trailGrid.length()) {
            trailGrid[i] = agentGrid[i].x;
        }
    }

    // Final barrier before rendering
    barrier();
    memoryBarrierBuffer();
    
    

    // Render trail grid to output texture - each thread renders one pixel
    vec4 color = vec4(0.0, 0.0, 0.0, 1.0);
    
    // Debug: Show agents as bright dots
    for (uint i = 0u; i < min(agentCount, 100u); i++) {
        vec2 agentPos = positions[i];
        float dist = distance(vec2(gl_GlobalInvocationID.xy), agentPos);
        if (dist < 2.0) {
            color = vec4(1.0, 0.0, 0.0, 1.0); // Red agent dots
        }
    }
    
    if (gl_GlobalInvocationID.x < uint(W) && gl_GlobalInvocationID.y < uint(H)) {
        int trailIndex = int(gl_GlobalInvocationID.x) + int(gl_GlobalInvocationID.y) * int(W);
        
        if (trailIndex >= 0 && trailIndex < trailGrid.length()) {
            float trailValue = trailGrid[trailIndex];
            if (trailValue > 0.0) {
                color = vec4(trailValue) ; // , trailValue * 0.1, trailValue * 0.2, 1.0);
            }
            
            // Reset agent grid for next frame
            agentGrid[trailIndex] = vec4(0.0);
        }
    }
    
    // Debug: Show center dot
    // vec2 center = vec2(rez / 2.0);
    // if (distance(vec2(gl_GlobalInvocationID.xy), center) < 3.0) {
    //     color = vec4(0.0, 1.0, 0.0, 1.0); // Green center dot
    // }
    // 
    // // Debug info in corner - show frame and agentCount
    // if (gl_GlobalInvocationID.x < 100u && gl_GlobalInvocationID.y < 20u) {
    //     float intensity = float(frame % 60) / 60.0; // Flashing based on frame
    //     color = vec4(intensity, intensity, 1.0, 1.0); // Blue flashing corner
    // }
    // 
    // // Debug: Show if any agent processing happened
    // uint debugAgentId = gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x;
    // if (debugAgentId < 100u && gl_GlobalInvocationID.x > 200u && gl_GlobalInvocationID.x < 250u && gl_GlobalInvocationID.y < 50u) {
    //     color = vec4(1.0, 1.0, 0.0, 1.0); // Yellow bar if agent processing happened
    // }

    // Audio spectrum visualization - logarithmic histogram moving from top to bottom
    const int MUSICAL_BINS = 352;
    vec2 screenPos = vec2(gl_GlobalInvocationID.xy);
    
    // Create scrolling audio history visualization in the top portion of the screen
    float historyHeight = H * 0.3;  // Use top 30% of screen for audio history
    
    if (screenPos.y < historyHeight) {
        // Scroll the backbuffer down by 1 pixel each frame
        if (screenPos.y > 0) {
            vec4 prevColor = imageLoad(backbuffer, ivec2(screenPos.x, screenPos.y - 1));
            imageStore(backbuffer, ivec2(screenPos.x, screenPos.y), prevColor * 0.95); // Fade over time
        }
        
        // Add new audio data to the top row
        if (screenPos.y == 0) {
            // Map screen X position to frequency bin (logarithmic distribution)
            float freqPos = screenPos.x / W;
            int binIndex = int(freqPos * float(MUSICAL_BINS));
            //binIndex = clamp(binIndex, 0, MUSICAL_BINS - 1);
            
            // Sample audio data
            vec4 audioSample = imageLoad(audioTexture, ivec2(binIndex, 0));
            float amplitude = audioSample.r;
            
            // Logarithmic scaling for better visualization
            float logAmplitude = log(1.0 + amplitude * 10.0) / log(11.0);
            
            // Create color based on frequency and amplitude
            vec3 spectrumColor = vec3(0.0);
            float hue = freqPos * 6.28318; // Map frequency to hue
            spectrumColor.r = sin(hue) * 0.5 + 0.5;
            spectrumColor.g = sin(hue + 2.094) * 0.5 + 0.5;  // 2π/3
            spectrumColor.b = sin(hue + 4.188) * 0.5 + 0.5;  // 4π/3
            
            // Set intensity based on amplitude
            vec4 audioColor = vec4(spectrumColor * logAmplitude * 2.0, 1.0);
            imageStore(backbuffer, ivec2(screenPos.x, 0), audioColor);
        }
        
        // Display the audio history from backbuffer
        vec4 historyColor = imageLoad(backbuffer, ivec2(screenPos.x, screenPos.y) + ivec2(0, -1));
        color += historyColor * 0.;
    }
    
    // Add the original physarum simulation with slight fade
    color += imageLoad(outputTexture, fragCoord) * vec4(.4);

    imageStore(outputTexture, fragCoord, color);
}
 
