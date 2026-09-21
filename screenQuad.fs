#version 430 core
out vec4 FragColor;

in vec2 TexCoords;

uniform sampler2D tex;
uniform float dim_factor;

void main()
{             
    vec3 texCol = texture(tex, TexCoords).rgb;      
    FragColor = vec4(texCol * dim_factor, 1.0);
}


