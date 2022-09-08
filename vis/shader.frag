#version 430

in vec4 fPosition;
in vec4 fColor;
in vec4 fNormal;
out vec4 FragColor;

uniform mat4 M;
uniform mat4 V;
uniform mat4 P;

layout(location=4) uniform int colorMode;


struct Material {
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    float shininess;
};
uniform Material material;

struct Light {
    vec4 position;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
    vec3 attenuation;
};
uniform Light light;

vec4 shading(vec3 LightPos_ec, vec3 vPosition_ec, vec3 vNormal_ec)
{
    vec3 N = normalize(vNormal_ec);
    vec3 L = LightPos_ec - vPosition_ec;
    float d = length(L);
    L = normalize(L);
    vec3 V = normalize(-vPosition_ec);
    vec3 R = reflect(-L, N);

    float fatt = min(1.0 / (light.attenuation.x + light.attenuation.y * d + light.attenuation.z * d * d), 1.0);
    float cos_theta = max(dot(L, N), 0.0);
    float cos_alpha = max(dot(R, V), 0.0);

    vec3 ambient = light.ambient * material.ambient;
    vec3 diffuse = light.diffuse * (material.diffuse * cos_theta);
    vec3 specular = light.specular * material.specular * pow(cos_alpha, material.shininess);
    vec3 I = ambient + fatt * (diffuse + specular);
    return vec4(I, 0);
}

void main()
{
    if(colorMode == 1)
    {
        FragColor = fColor;
    }

    // Phong shading
    else
    {
        // convert the coordinate of vectors according to the eye coordinate
        mat4 VM = V * M;
        mat4 U = transpose(inverse(VM));
        vec3 vNormal_ec = vec3(normalize(U * fNormal));
        vec3 fPosition_ec = vec3(VM * fPosition);
        vec3 LightPos_ec = vec3(V * light.position);
        
	    // comopute the color based on the interpolated normal vectors
	    FragColor = shading(LightPos_ec, fPosition_ec, vNormal_ec);
    }
}
