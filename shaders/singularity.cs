#version 430 core

layout (local_size_x = 10, local_size_y = 10, local_size_z = 1) in;

// ----------------------------------------------------------------------------
//
// uniforms
//
// ----------------------------------------------------------------------------

layout(rgba32f, binding = 0) uniform image2D imgOutput;
layout(rgba8, binding = 2) uniform image2D cameraInput;

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

    uv -= .5;
    uv *= 2.;
    uv.x *= 1.5;

	value.x = mod(float(texelCoord.x) + t * speed, width) / (gl_NumWorkGroups.x * gl_WorkGroupSize.x);
	// value.z = float(texelCoord.y)/(gl_NumWorkGroups.y*gl_WorkGroupSize.y);

    float c = abs(circle(uv, .9)) < 0.005? 1. : 0.;

    float r =.10;
    float w = 2.3;
    float radar = uv.x * sin(t) + uv.y * cos(t);    
    radar = 1.;
    value.x = mod(float(texelCoord.y ) / (circle(uv , r) * 1.9 )  + t * speed, width) < w / radar ? 1. : 0.;
    value.z *= sin(t);
    //value.x *= cos(t);
    value.x /= pow(circle(uv, r), 26.) ;
    value.xyz = vec3(value.x);

    //value = vec4(1.);
	imageStore(imgOutput, texelCoord, value);
}

