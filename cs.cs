#version 430 core

#define SHIFT(X) (.5 * (X + 1))
#define TON(X) (2.* (X -.5))

layout (local_size_x = 10, local_size_y = 10, local_size_z = 1) in;

// ----------------------------------------------------------------------------
//
// uniforms
//
// ----------------------------------------------------------------------------

layout(rgba32f, binding = 0) uniform image2D imgOutput;

layout (location = 0) uniform float t;                 /** Time */

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
	float speed = 100;
	// the width of the texture
	float width = 100;
    
    vec2 uv = vec2(texelCoord.xy) / (vec2(gl_NumWorkGroups.xy) * vec2(gl_WorkGroupSize.xy));
    value = vec4(uv.x, 0., uv.y, 1.);
	imageStore(imgOutput, texelCoord, value);
}

