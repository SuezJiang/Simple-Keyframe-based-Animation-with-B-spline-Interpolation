#version 330 core
out vec4 FragColor;


struct DirLight {
    vec3 direction;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

struct PointLight {
    vec3 position;
    float constant;
    float linear;
    float quadratic;
    vec3 ambient;
    vec3 diffuse;
    vec3 specular;
};

struct Material {
    sampler2D diffuse;
    sampler2D specular;
    float shininess;
};

#define NR_POINT_LIGHTS 4

in VS_OUT {
    vec3 FragPos;
    vec3 Normal;
    vec2 TexCoords;
    vec3 TangentViewPos;
    vec3 TangentFragPos;
} fs_in;
in vec3 TangentLightPos[4];
in vec3 TangentdirLightDir;

uniform vec3 viewPos;
uniform DirLight dirLight;
uniform PointLight pointLights[NR_POINT_LIGHTS];
uniform Material mat;
uniform sampler2D normalMap;
// a switch
uniform bool useNormalMap;

// function prototypes
vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir);
vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir, vec3 realLightPos);

void main()
{    
   vec3 normal;
   vec3 viewDir;
   vec3 realFragPos;
   if(useNormalMap){
       // use transformed data
       normal = texture(normalMap, fs_in.TexCoords).rgb;
       normal = normalize(2 * normal - 1);
       viewDir = normalize(fs_in.TangentViewPos - fs_in.TangentFragPos);
       realFragPos = fs_in.TangentFragPos;
   }else{
       // original data
       normal = normalize(fs_in.Normal);
       viewDir = normalize(viewPos - fs_in.FragPos);
       realFragPos = fs_in.FragPos;
   }
   vec3 finalEffect = vec3(0);
   for(int i = 0; i < NR_POINT_LIGHTS; i++){
      if(useNormalMap){
          finalEffect += CalcPointLight(pointLights[i],normal,realFragPos,viewDir,TangentLightPos[i]);
	  }else{
          finalEffect += CalcPointLight(pointLights[i],normal,realFragPos,viewDir,pointLights[i].position);
	  }
   } 
   finalEffect += CalcDirLight(dirLight, normal, viewDir);
   FragColor = vec4(finalEffect ,0);
    
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//  Fill these functions. And use them in main.                                                                //
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// calculates the color when using a directional light.

vec3 CalcDirLight(DirLight light, vec3 normal, vec3 viewDir)

{
    vec3 lightDir;
    if(useNormalMap){
        lightDir = normalize(-TangentdirLightDir);
	}else{
        lightDir = normalize(-light.direction);
	}
    vec3 ambient;
    ambient = light.ambient * vec3(texture(mat.diffuse, fs_in.TexCoords));
    vec3 diffuse;
    float diffStrength = max(dot(normal,lightDir), 0);
    diffuse = light.diffuse * diffStrength * vec3(texture(mat.diffuse, fs_in.TexCoords));
    vec3 specular;
    vec3 reflectDir = reflect(-lightDir, normal);
    float specStrength = pow(max(dot(viewDir,reflectDir), 0), 16);
    specular = light.specular * specStrength * vec3(texture(mat.specular, fs_in.TexCoords));
    return (ambient + diffuse + specular);
}



// calculates the color when using a point light.

vec3 CalcPointLight(PointLight light, vec3 normal, vec3 fragPos, vec3 viewDir, vec3 realLightPos)
{
    vec3 lightDir = normalize(realLightPos - fragPos);
    vec3 ambient;
    ambient = light.ambient * vec3(texture(mat.diffuse, fs_in.TexCoords));
    vec3 diffuse;
    float diffStrength = max(dot(normal, lightDir), 0.0);
    diffuse = light.diffuse * diffStrength * vec3(texture(mat.diffuse, fs_in.TexCoords));
    vec3 specular;
    vec3 reflectDir = reflect(-lightDir,normal);
    float specStrength = pow(max(dot(viewDir,reflectDir),0.0),16);
    specular = light.specular * specStrength * vec3(texture(mat.specular, fs_in.TexCoords));
    float attenuation;
    float distance = length(realLightPos - fragPos);
    attenuation = 1/ (light.constant + light.linear*distance + light.quadratic*pow(distance,2));
    return (ambient + diffuse + specular)*attenuation;
}


