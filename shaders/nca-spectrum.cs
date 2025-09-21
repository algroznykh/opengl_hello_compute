#version 430 core

layout (local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

// Textures and images - matching compute.cpp glBindImageTexture calls
layout(rgba32f, binding = 0) uniform image2D outputTexture;          // glBindImageTexture(0, ...)
layout(r32f, binding = 1) uniform image2D audioTexture;              // glBindImageTexture(1, ...)  
layout(rgba8, binding = 2) uniform image2D camTexture;               // glBindImageTexture(2, ...)
layout(rgba32f, binding = 3) uniform image2D backbuffer;

// Uniforms buffer - binding 4 (avoiding conflict with image bindings)
layout(std140, binding = 4) uniform Uniforms {
    mat3 kernel;
    uint filter_type;
    uint frame;
};

// Storage buffer for states - binding 5 (avoiding conflict with image bindings)
layout(std430, binding = 5) restrict buffer StateBuffer {
    float states[][12];
};

// Constants
const uint N = 12u;
const float S = 5000.0;
const int[12] B = int[12](-172,-203,-249,333,68,356,219,293,228,308,-259,70);

// Weight matrix W (48x12)
const int[48][12] W = int[48][12](
    int[12](-570,-6,-50,69,82,37,125,64,78,191,-157,-20),
    int[12](-94,-501,-16,95,-12,-97,184,99,21,176,-63,-143),
    int[12](43,-73,-510,167,166,44,136,116,18,38,-137,-65),
    int[12](-11,-37,-94,-778,-172,-47,57,-153,157,79,-99,168),
    int[12](-50,-43,-94,226,-1208,307,-1,-62,-44,-30,-52,25),
    int[12](-59,-34,-67,15,-131,-1093,-24,64,-25,-82,58,11),
    int[12](2,-33,18,-47,-64,200,-934,188,130,48,1,45),
    int[12](-280,-291,-321,436,153,-114,-107,-1008,-46,102,-139,90),
    int[12](-100,-56,-33,13,18,-19,-99,223,-664,96,23,-50),
    int[12](-130,-125,-27,132,-91,-36,77,-52,-341,-1060,-496,56),
    int[12](-24,-21,-9,-253,135,-357,-165,132,-156,232,-763,-76),
    int[12](29,70,41,-188,-114,202,-29,41,160,73,110,-799),
    int[12](137,-77,-43,-20,-9,-15,55,-18,1,-18,-79,39),
    int[12](30,178,-86,-71,17,16,48,-17,-9,24,-59,50),
    int[12](-23,32,272,-55,-6,-10,34,-26,5,-10,-13,40),
    int[12](309,337,363,422,-52,-83,58,82,4,47,-37,-55),
    int[12](-26,-22,-38,54,-349,-204,18,28,-6,-28,-1,-39),
    int[12](-128,-127,-129,109,-54,250,-10,-20,-17,24,35,30),
    int[12](10,5,14,24,17,107,-79,20,-7,-4,22,5),
    int[12](1,-1,-8,-42,19,-92,45,-113,-43,52,-29,26),
    int[12](-126,-143,-133,-22,-4,36,-31,74,153,-157,-15,73),
    int[12](-60,-50,-39,142,21,-60,-55,148,3,-232,173,-118),
    int[12](-106,-95,-107,129,-13,8,-26,-120,-93,111,1,53),
    int[12](-12,-28,-21,-108,72,3,16,186,117,-134,149,228),
    int[12](-9,-173,-89,-235,209,-33,-207,43,-269,-252,75,-447),
    int[12](-194,11,-74,-230,262,22,-225,47,-225,-235,5,-323),
    int[12](-261,-205,13,-212,255,-51,-244,87,-271,-181,-3,-486),
    int[12](20,24,5,-245,-146,61,-126,-75,162,30,61,8),
    int[12](11,25,-21,-85,-532,-111,-31,-120,13,39,104,-36),
    int[12](-49,-45,-41,145,-61,-529,47,115,36,-126,17,-77),
    int[12](90,83,35,-99,-159,53,-136,2,63,215,351,-31),
    int[12](-97,-87,-97,110,-8,-11,-8,-39,-71,-61,84,31),
    int[12](-19,-1,14,184,-26,15,2,9,-13,-1,102,-90),
    int[12](108,118,73,-255,-134,7,-62,-124,77,128,-53,5),
    int[12](104,79,114,-282,-124,168,52,-300,-167,-190,-335,324),
    int[12](-59,-60,-58,-43,73,-135,28,-19,-64,-1,-99,456),
    int[12](-3,50,34,-30,-30,36,23,-15,27,10,-2,40),
    int[12](35,-11,26,3,-26,21,22,-49,30,4,-17,48),
    int[12](-4,-16,-39,95,35,-1,14,-12,-14,-14,-10,15),
    int[12](-76,-45,-10,93,228,34,56,92,-115,-148,3,-64),
    int[12](62,57,54,-28,71,78,28,-106,84,90,-27,-14),
    int[12](94,89,65,-134,-199,182,67,-135,4,88,-11,87),
    int[12](-59,-39,-7,271,64,-33,-84,39,-175,-116,-103,-153),
    int[12](55,46,16,-49,-163,43,17,-12,109,43,171,-48),
    int[12](140,118,115,-245,1,24,163,-200,-21,-81,-34,177),
    int[12](-6,-4,2,68,102,-69,71,61,88,-27,45,-56),
    int[12](75,74,60,63,102,-93,39,45,130,115,145,-61),
    int[12](-81,-72,-63,125,14,-68,-86,85,-69,-152,16,-171)
);

// Simulation size
uint SW = imageSize(outputTexture).x;
uint SH = imageSize(outputTexture).y;

// Global variable for current index
ivec2 current_index;

// Helper functions
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
    uint x = (uint(current_index.x + dx) + SW) % SW;
    uint y = (uint(current_index.y + dy) + SH) % SH;
    return get_xyc(x, y, c);
}

float lap(uint c) {
    return R(1,1,c) + R(1,-1,c) + R(-1,1,c) + R(-1,-1,c) 
        + 2.0 * (R(0,1,c) + R(0,-1,c) + R(1,0,c) + R(-1,0,c)) - 12.0*R(0,0,c);
}

float sobx(uint c) {
    return R(-1, 1, c) + R(-1, 0, c)*2.0 + R(-1,-1, c)
          -R( 1, 1, c) - R( 1, 0, c)*2.0 - R( 1,-1, c);
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

    // Do 1x1 conv
    float[12] y;
    for (uint c = 0u; c < N; c++) {
        float val = float(B[c]);

        for (uint i = 0u; i < 48u; i++) {
            val += hs[i] * float(W[i][c]);
        }
        y[c] = xs[c] + val / S;
        y[c] = clamp(y[c], -1.5, 1.5);
    }

    if (abs(y[4]) < 0.01) {
        return xs;
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
    ivec2 screen_size = imageSize(outputTexture);
    ivec2 fragCoord = ivec2(gl_GlobalInvocationID.xy);
	ivec2 texelCoord = ivec2(gl_GlobalInvocationID.xy);
    ivec2 imgSize = screen_size;
    vec2 uv = vec2(texelCoord.xy) / vec2(imgSize);
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
    vec4 cameraColor = imageLoad(camTexture, texelCoord);

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
    
    vec4 back = imageLoad(backbuffer, texelCoord + ivec2(1, 0));
    //vec4 back = sampleBilinear(backbuffer, rotatedUV);
    //back *= angle < .9 ? 1.: 1. - angle;
    back *= angle < 1. ? 1.: 1. - angle;
    back += vec4(spectrum, 1.);
    imageStore(backbuffer, texelCoord, back);

    finalColor += sampleBilinear(backbuffer, polar.yx).rgb;
    
    vec4 value = vec4(finalColor, 1.0);

    vec4 tex = camlap(fragCoord);
    if (gl_GlobalInvocationID.x >= uint(screen_size.x) || gl_GlobalInvocationID.y >= uint(screen_size.y)) { 
        return; 
    }

    if (gl_GlobalInvocationID.x < SW && gl_GlobalInvocationID.y < SH) { 
        current_index = ivec2(int(gl_GlobalInvocationID.x), int(gl_GlobalInvocationID.y));

        // Initial state
        if (frame == 1u) {
            float[12] init_s;
            for (uint s = 0u; s < N; s++) {
                float a = 0.01;
                float rand = fract(sin(float((gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * SW) * (s+1)) / float(SW)) * 353348.5453123) + a;
                init_s[s] = floor(rand);
            }
            set_xy(uint(current_index.x), uint(current_index.y), init_s);
            return;
        }

        float[12] ps = float[12](
            lap(0u) + sin(tex.r),
            lap(1u) + sin(tex.g),
            lap(2u) + sin(tex.b),
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

        set_xy(uint(current_index.x), uint(current_index.y), state);
    }

    // Rescale buffer 
    uint idxs = uint(float(gl_GlobalInvocationID.x) / float(screen_size.x) * float(SW));
    uint idys = uint(float(gl_GlobalInvocationID.y) / float(screen_size.y) * float(SH));

    // Output to screen
    float[12] states_out = get_xy(idxs, idys);
    vec4 xrgb = vec4(states_out[0], states_out[1], states_out[2], states_out[3]) + 0.5;

    xrgb *= xrgb * value ;

    xrgb = value;

    xrgb.x = step(polar.x, 0.99);
    //xrgb = back;
    xrgb.z = spectrum.x;



    imageStore(outputTexture, fragCoord, xrgb);
}
