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

uint D=1; // dilation

// Simulation size
uint FACTOR = 4;
uint SW = imageSize(outputTexture).x / FACTOR;
uint SH = imageSize(outputTexture).y / FACTOR;

// Global variable for current index
ivec2 current_index;

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
    vec2 uv = vec2(texelCoord.xy) / vec2(SW, SH);
    vec2 centered = uv - 0.5;
    centered *= 2.;
    vec2 ratio = centered * vec2(float(SW)/float(SH), 1.);
    centered = ratio;

    float radius = length(centered);
    float angle = atan(centered.y, centered.x);
    float s = time / 2.5;
    
    // Normalize angle to [0, 1] range
    angle = (angle + 3.14159) / (2.0 * 3.14159);
    angle = 1. - angle;
    angle = angle + .75;
    angle = fract(angle); // loop at midnight

    vec2 polar = vec2(radius, angle) ;

    // Sample camera input
    vec4 cameraColor = imageLoad(camTexture, texelCoord);

    // Get audio data - map angle to musical scale bins for circular spectrogram
    float MUSICAL_BINS = 88.0 * 4.; // Fixed number of musical scale bins
    //int audioIndex = int((1.-radius) * MUSICAL_BINS);
    // SPECTRAL
    int audioIndex = int(( 1. - radius / 1.) * MUSICAL_BINS);
    audioIndex = clamp(audioIndex, 0, int(MUSICAL_BINS)-1);
    
    int SPECTRUM_SHIFT = 0;
    //audioIndex = int((1. - radius) * SH) * 8.;
    float audioMagnitude = imageLoad(audioTexture, ivec2(audioIndex + SPECTRUM_SHIFT, 0)).r;
    audioMagnitude = clamp(audioMagnitude * 200.0, 0.0, 1.0); // Amplify and clamp
    
    vec3 color = vec3(audioMagnitude);
    // Use radius for intensity instead of x coordinate
    //float intensity = angle < 0.01 ? 1. : 0.;
    float intensity = abs(angle - fract(s)) < 0.01 ? 1. : 0.;
    //intensity *= 1. - radius;
    vec3 spectrum = color * intensity * 2.;
    vec3 finalColor ;
    
    
    //vec4 back = imageLoad(backbuffer, texelCoord + ivec2(1, 0));
    vec4 back = imageLoad(backbuffer, texelCoord); 
    //vec4 back = sampleBilinear(backbuffer, rotatedUV);
    //back *= angle < .9 ? 1.: 1. - angle;
    //back *= angle < 1. ? 1.: 1. - angle;
    //back += angle + fract(time) < 1.? vec4(spectrum, 1.) : vec4(0.);
    //back += vec4(spectrum, 1.);
    back = max(back,vec4(spectrum,1.));

    back *= .995;
    //back += vec4(spectrum, 1.) * vec4(.99, .98, .97, 1.);
    //back *= vec4(.9995, .99, .9995, 1.);
    imageStore(backbuffer, texelCoord, back);

    //finalColor +=1. *   sampleLoG(backbuffer, uv).rgb;
    //finalColor *= .15 * sampleBilinear(backbuffer, uv).rgb;
    finalColor += back.rgb;

    //finalColor *= length(finalColor) > .5 ? 1. : 0.;
    
    vec4 value = vec4(finalColor, 1.0);
    value *= 1. - radius;

    vec4 tex = camlap(fragCoord);
    if (gl_GlobalInvocationID.x >= uint(screen_size.x) || gl_GlobalInvocationID.y >= uint(screen_size.y)) { 
        return; 
    }

    
    //float dial = step(circle(ratio, .75), .0);
    float dial =  .05 - circle(ratio, .75);
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
    dial = dial + ccc;
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
            sobx(4u),// - tex.r * 2.,
            sobx(5u), 
            sobx(6u), 
            sobx(7u),
            soby(8u),// - tex.b * 2.,
            soby(9u),
            soby(10u),
            soby(11u)
        );
        
        // Update state
        float[12] xs = get_xy(uint(current_index.x), uint(current_index.y));    
        float[12] state = update(xs, ps);
        vec4 scaled_back = imageLoad(backbuffer, texelCoord * int(FACTOR)); 
        scaled_back = back;

        if (frame < 100) {
        return;
        }
       
        // disturb states
        for (uint s = 0u; s < N; s++) {
            //state[s] *= length(scaled_back) > 1.? 1.05 : 0.;
            state[s] *= dial.x;
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
    vec4 xrgb = vec4(states_out[0], states_out[1], states_out[2], states_out[3]);
    

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


    imageStore(outputTexture, fragCoord, xrgb);
    // interpolation
    vec4 interpolated = interp(fragCoord, int(FACTOR-1));
    //interpolated.x = dial.x;
    imageStore(outputTexture, fragCoord, interpolated);


}
 
