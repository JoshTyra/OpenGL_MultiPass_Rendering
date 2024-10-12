#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <vector>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include "Camera.h"
#include <random>
#include <map>
#include "FileSystemUtils.h"

// Asset Importer
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

void APIENTRY MessageCallback(GLenum source,
    GLenum type,
    GLuint id,
    GLenum severity,
    GLsizei length,
    const GLchar* message,
    const void* userParam)
{
    std::cerr << "GL CALLBACK: " << (type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : "")
        << " type = " << type
        << ", severity = " << severity
        << ", message = " << message << std::endl;
}

// Constants and global variables
const int WIDTH = 2560;
const int HEIGHT = 1080;
float lastX = WIDTH / 2.0f;
float lastY = HEIGHT / 2.0f;
bool firstMouse = true;
float deltaTime = 0.0f; // Time between current frame and last frame
float lastFrame = 0.0f; // Time of last frame
double previousTime = 0.0;
int frameCount = 0;

Camera camera(glm::vec3(0.0f, 5.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), -180.0f, 0.0f, 6.0f, 0.1f, 45.0f, 0.1f, 500.0f);

const char* vertexShaderSource = R"(
    #version 430 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;
    layout (location = 2) in vec2 aTexCoords;
    layout (location = 3) in vec2 aLightmapTexCoords;

    out vec2 TexCoords;
    out vec2 LightmapTexCoords;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main() {
        TexCoords = aTexCoords;
        LightmapTexCoords = aLightmapTexCoords; // Pass lightmap UVs
        gl_Position = projection * view * model * vec4(aPos, 1.0);
    }
)";

const char* fragmentShaderSource = R"(
    #version 430 core
    out vec4 FragColor;

    in vec2 TexCoords;
    in vec2 LightmapTexCoords;

    uniform sampler2D diffuseTexture;
    uniform sampler2D lightmapTexture;
    uniform int debugMode; // 0: Combined, 1: Diffuse, 2: Lightmap
    uniform float blendFactor; // Control lightmap influence

    // Helper function to convert from sRGB to linear color space
    vec3 sRGBtoLinear(vec3 color) {
        return pow(color, vec3(2.2)); // Approximate sRGB to linear conversion
    }

    // Helper function to convert from linear to sRGB color space
    vec3 LinearTosRGB(vec3 color) {
        return pow(color, vec3(1.0 / 2.2)); // Approximate linear to sRGB conversion
    }

    // Helper function to adjust contrast
    vec3 adjustContrast(vec3 color, float contrast) {
        return clamp((color - 0.5) * contrast + 0.5, 0.0, 1.0);
    }

    void main() {
        // Sample diffuse and lightmap textures (assumed to be in sRGB space)
        vec4 diffuseColor = texture(diffuseTexture, TexCoords); 
        vec4 lightmapColor = texture(lightmapTexture, LightmapTexCoords);

        // Convert both diffuse and lightmap colors from sRGB to linear space
        vec3 diffuseLinear = sRGBtoLinear(diffuseColor.rgb);
    
        // Increase lightmap contrast to emphasize shadows
        vec3 lightmapLinear = adjustContrast(sRGBtoLinear(lightmapColor.rgb), 1.5); 

        // Use multiplicative blending to emphasize shadows
        vec3 combinedLinear = mix(diffuseLinear, diffuseLinear * lightmapLinear, blendFactor);

        // Convert the final color back to sRGB space for display
        vec3 finalColor = LinearTosRGB(combinedLinear);

        // Debug mode switching
        if (debugMode == 1) {
            finalColor = diffuseColor.rgb; // Show diffuse texture only (still in sRGB)
        } else if (debugMode == 2) {
            finalColor = lightmapColor.rgb; // Show lightmap texture only (still in sRGB)
        }

        // Output the final color with the diffuse texture's alpha channel
        FragColor = vec4(finalColor, diffuseColor.a); 
    }
)";

// New Vertex Shader Source
const char* newVertexShaderSource = R"(
    #version 430 core
    layout (location = 0) in vec3 aPos;
    layout (location = 1) in vec3 aNormal;
    layout (location = 2) in vec2 aTexCoords;
    layout (location = 4) in vec3 aTangent;
    layout (location = 5) in vec3 aBitangent;

    out vec2 TexCoords;
    out vec3 FragPos;
    out vec3 T;
    out vec3 B;
    out vec3 N;

    uniform mat4 model;
    uniform mat4 view;
    uniform mat4 projection;

    void main() {
        TexCoords = aTexCoords;
        FragPos = vec3(model * vec4(aPos, 1.0));
        mat3 normalMatrix = transpose(inverse(mat3(model)));
        T = normalize(normalMatrix * aTangent);
        B = normalize(normalMatrix * aBitangent);
        N = normalize(normalMatrix * aNormal);
        gl_Position = projection * view * vec4(FragPos, 1.0);
    }
)";

const char* newFragmentShaderSource = R"(
    #version 430 core
    out vec4 FragColor;

    in vec2 TexCoords;
    in vec3 FragPos;
    in vec3 T;
    in vec3 B;
    in vec3 N;

    uniform sampler2D diffuseTexture;
    uniform sampler2D normalMap;
    uniform int debugMode;
    uniform vec3 lightDir;
    uniform vec3 viewPos;
    uniform float shininess = 32.0f;
    uniform vec3 specularColor = vec3(0.35f, 0.35f, 0.35f);

    void main() {
        // Obtain normal from normal map in range [0,1]
        vec3 normal = texture(normalMap, TexCoords).rgb;
        normal = normalize(normal * 2.0 - 1.0);

        // Construct TBN matrix
        mat3 TBN = mat3(T, B, N);
        vec3 worldNormal = normalize(TBN * normal);

        // Compute lighting direction
        vec3 lightDirection = normalize(-lightDir);

        // Compute per-pixel diffuse term using normal mapping
        float diff = max(dot(worldNormal, lightDirection), 0.0);

        // Sample the diffuse texture for the base color
        vec3 baseColor = texture(diffuseTexture, TexCoords).rgb;

        // Compute the diffuse component
        vec3 diffuse = diff * baseColor;

        // Blinn-Phong specular calculation
        vec3 viewDir = normalize(viewPos - FragPos);
        vec3 halfDir = normalize(viewDir + lightDirection);
        float specAngle = max(dot(worldNormal, halfDir), 0.0);
        float spec = pow(specAngle, shininess);

        // Sample the alpha channel from the diffuse texture to get the specular mask
        float specularMask = texture(diffuseTexture, TexCoords).a;

        // Modulate the specular component with the specular mask
        vec3 specular = spec * specularColor * specularMask;

        // Combine diffuse and specular components
        vec3 finalColor = diffuse + specular;

        // Debug modes
        if (debugMode == 1) {
            finalColor = baseColor; // Base color only
        } else if (debugMode == 2) {
            finalColor = normal * 0.5 + 0.5; // Visualize normal map (tangent space)
        } else if (debugMode == 3) {
            finalColor = diffuse; // Diffuse component only
        } else if (debugMode == 4) {
            finalColor = specular; // Specular component only
        } else if (debugMode == 5) {
            finalColor = vec3(specularMask); // Visualize specular mask
        } else if (debugMode == 6) {
            finalColor = worldNormal * 0.5 + 0.5; // Visualize world-space normal
        }

        FragColor = vec4(finalColor, 1.0);
    }
)";

const char* quadVertexShaderSource = R"(
    #version 430 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aTexCoords;

    out vec2 TexCoords;

    void main() {
        TexCoords = aTexCoords;
        gl_Position = vec4(aPos, 0.0, 1.0);
    }
)";

const char* combinedQuadFragmentShaderSource = R"(
    #version 430 core
    out vec4 FragColor;
    in vec2 TexCoords;

    uniform sampler2D firstPassTexture;
    uniform sampler2D secondPassTexture;

    void main() {
        // Sample colors from both textures
        vec3 color1 = texture(firstPassTexture, TexCoords).rgb;
        vec3 color2 = texture(secondPassTexture, TexCoords).rgb;

        // Convert to linear space for accurate blending
        vec3 linearColor1 = pow(color1, vec3(2.2));
        vec3 linearColor2 = pow(color2, vec3(2.2));

        float diffuseFactor = 1.5;
        float specularFactor = 1.0;

        vec3 combinedLinearColor = linearColor1 * diffuseFactor + linearColor2 * specularFactor;

        // Apply tone mapping or clamp the color if necessary
        //combinedLinearColor = clamp(combinedLinearColor, 0.0, 1.0);

        // Convert back to sRGB for display
        vec3 finalColor = pow(combinedLinearColor, vec3(1.0 / 2.2));

        FragColor = vec4(finalColor, 1.0);
    }
)";

void processInput(GLFWwindow* window) {
    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.processKeyboardInput(GLFW_KEY_W, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.processKeyboardInput(GLFW_KEY_S, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.processKeyboardInput(GLFW_KEY_A, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.processKeyboardInput(GLFW_KEY_D, deltaTime);
}

void mouseCallback(GLFWwindow* window, double xpos, double ypos) {
    static bool firstMouse = true;
    static float lastX = WIDTH / 2.0f;
    static float lastY = HEIGHT / 2.0f;

    if (firstMouse) {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // Reversed since y-coordinates range from bottom to top
    lastX = xpos;
    lastY = ypos;

    camera.processMouseMovement(xoffset, yoffset);
}

void scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    camera.processMouseScroll(static_cast<float>(yoffset));
}

// Define this mapping globally or pass it to your loadModel function
std::map<std::string, std::string> materialNormalMapPaths = {
    {"example_tutorial_ground", "textures/metal flat generic bump.png"},
    {"example_tutorial_metal", "textures/metal flat generic bump.png"},
    {"example_tutorial_metal_floor", "textures/metal flat generic bump.png"},
    {"example_tutorial_plate_floor", "textures/metal plate floor bump.png"},
    {"example_tutorial_panels", "textures/metal flat generic bump.png"},
    {"boulder_grey", "textures/metal flat generic bump.png"}
};

// Utility function to load textures using stb_image or similar
GLuint loadTextureFromFile(const char* path, const std::string& directory);
GLuint lightmapTexture;

GLuint createFlatNormalMap() {
    // Create a 1x1 texture with RGB value (0.5, 0.5, 1.0)
    unsigned char flatNormalData[3] = { 128, 128, 255 }; // (0.5 * 255, 0.5 * 255, 1.0 * 255)

    GLuint textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_2D, textureID);

    // Generate the texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 1, 1, 0, GL_RGB, GL_UNSIGNED_BYTE, flatNormalData);

    // Set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    return textureID;
}

std::string getFilenameFromPath(const std::string& path) {
    size_t pos = path.find_last_of("/\\");
    if (pos != std::string::npos)
        return path.substr(pos + 1);
    else
        return path;
}

struct Vertex {
    glm::vec3 Position;
    glm::vec3 Normal;
    glm::vec2 TexCoords;         // For diffuse texture
    glm::vec2 LightmapTexCoords; // For lightmap texture
    glm::vec3 Tangent;
    glm::vec3 Bitangent;
};

struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
    mutable unsigned int VAO;  // Mark as mutable to allow modification in const functions
    GLuint diffuseTexture;  // Store diffuse texture ID
    GLuint normalMapTexture;

    // Updated constructor
    Mesh(std::vector<Vertex> vertices, std::vector<unsigned int> indices, GLuint diffuseTexture, GLuint normalMapTexture)
        : vertices(vertices), indices(indices), diffuseTexture(diffuseTexture), normalMapTexture(normalMapTexture) {
        setupMesh();
    }

    void setupMesh() const {
        // Set up the VAO, VBO, and EBO as before
        glGenVertexArrays(1, &VAO);
        glBindVertexArray(VAO);

        unsigned int VBO, EBO;
        glGenBuffers(1, &VBO);
        glGenBuffers(1, &EBO);

        glBindBuffer(GL_ARRAY_BUFFER, VBO);
        glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);

        // Vertex Positions
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)0);
        // Vertex Normals
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Normal));
        // Vertex Texture Coords (Diffuse)
        glEnableVertexAttribArray(2);
        glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, TexCoords));
        // Lightmap Texture Coords
        glEnableVertexAttribArray(3);
        glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, LightmapTexCoords));
        // Tangents
        glEnableVertexAttribArray(4);
        glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Tangent));
        // Bitangents
        glEnableVertexAttribArray(5);
        glVertexAttribPointer(5, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, Bitangent));

        glBindVertexArray(0);
    }

    void Draw(GLuint shaderProgram) const {
        // Bind diffuse texture
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, diffuseTexture);
        glUniform1i(glGetUniformLocation(shaderProgram, "diffuseTexture"), 0);

        // Bind normal map texture
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, normalMapTexture);
        glUniform1i(glGetUniformLocation(shaderProgram, "normalMap"), 1);

        // Bind VAO and draw the mesh
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }

    // Renders the meshes in the scene without diffuse textures for the 2nd pass
    void DrawWithoutDiffuse(GLuint shaderProgram) const {
        // Bind normal map texture
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, normalMapTexture);
        glUniform1i(glGetUniformLocation(shaderProgram, "normalMap"), 1);

        // Bind VAO and draw the mesh
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
        glBindVertexArray(0);
    }
};

std::vector<Mesh> loadModel(const std::string& path) {
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(path,
        aiProcess_Triangulate | aiProcess_FlipUVs |
        aiProcess_GenSmoothNormals | aiProcess_JoinIdenticalVertices |
        aiProcess_CalcTangentSpace);

    if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
        std::cerr << "ERROR::ASSIMP::" << importer.GetErrorString() << std::endl;
        return {};
    }

    // Create the default flat normal map texture once
    static GLuint flatNormalMap = createFlatNormalMap();

    std::vector<Mesh> meshes;

    for (unsigned int i = 0; i < scene->mNumMeshes; i++) {
        aiMesh* mesh = scene->mMeshes[i];
        std::vector<Vertex> vertices;
        std::vector<unsigned int> indices;
        GLuint diffuseTexture = 0;
        GLuint normalMapTexture = 0;
        GLuint lightmapTexture = 0;

        // Process vertices and indices
        for (unsigned int j = 0; j < mesh->mNumVertices; j++) {
            Vertex vertex;
            vertex.Position = glm::vec3(mesh->mVertices[j].x, mesh->mVertices[j].y, mesh->mVertices[j].z);
            vertex.Normal = glm::vec3(mesh->mNormals[j].x, mesh->mNormals[j].y, mesh->mNormals[j].z);

            // First UV channel for diffuse texture
            if (mesh->mTextureCoords[0]) {
                vertex.TexCoords = glm::vec2(mesh->mTextureCoords[0][j].x, mesh->mTextureCoords[0][j].y);
            }
            else {
                vertex.TexCoords = glm::vec2(0.0f, 0.0f);
            }

            // Second UV channel for lightmap texture
            if (mesh->mTextureCoords[1]) {
                vertex.LightmapTexCoords = glm::vec2(mesh->mTextureCoords[1][j].x, mesh->mTextureCoords[1][j].y);
            }
            else {
                // If the second UV channel doesn't exist, we can default to the first one or set to zero
                vertex.LightmapTexCoords = glm::vec2(0.0f, 0.0f);
            }

            // Get tangent and bitangent from Assimp
            if (mesh->HasTangentsAndBitangents()) {
                vertex.Tangent = glm::vec3(mesh->mTangents[j].x, mesh->mTangents[j].y, mesh->mTangents[j].z);
                vertex.Bitangent = glm::vec3(mesh->mBitangents[j].x, mesh->mBitangents[j].y, mesh->mBitangents[j].z);
            }
            else {
                vertex.Tangent = glm::vec3(0.0f);
                vertex.Bitangent = glm::vec3(0.0f);
            }

            vertices.push_back(vertex);
        }

        // Collect indices
        for (unsigned int j = 0; j < mesh->mNumFaces; j++) {
            aiFace face = mesh->mFaces[j];
            for (unsigned int k = 0; k < face.mNumIndices; k++) {
                indices.push_back(face.mIndices[k]);
            }
        }

        // Load the material
        aiMaterial* material = scene->mMaterials[mesh->mMaterialIndex];

        aiString name;
        material->Get(AI_MATKEY_NAME, name);
        std::string matName(name.C_Str());

        // Print the material name for debugging
        std::cout << "Material Name: " << matName << std::endl;

        // Load diffuse texture
        if (material->GetTextureCount(aiTextureType_DIFFUSE) > 0) {
            aiString str;
            material->GetTexture(aiTextureType_DIFFUSE, 0, &str);
            std::string textureFilename = getFilenameFromPath(str.C_Str());
            std::string texturePath = FileSystemUtils::getAssetFilePath("textures/" + textureFilename);
            diffuseTexture = loadTextureFromFile(texturePath.c_str(), "");
        }

        // Check if the material has a normal map specified in the mapping
        auto it = materialNormalMapPaths.find(matName);
        if (it != materialNormalMapPaths.end()) {
            // Use FileSystemUtils to get the full path
            std::string normalMapPath = FileSystemUtils::getAssetFilePath(it->second);
            normalMapTexture = loadTextureFromFile(normalMapPath.c_str(), "");

            // Print the normal map path for debugging
            std::cout << "Using Normal Map: " << normalMapPath << std::endl;
        }
        else if (material->GetTextureCount(aiTextureType_NORMALS) > 0) {
            // Existing code to load normal map from the model file
            aiString str;
            material->GetTexture(aiTextureType_NORMALS, 0, &str);
            std::string textureFilename = getFilenameFromPath(str.C_Str());
            std::string texturePath = FileSystemUtils::getAssetFilePath("textures/" + textureFilename);
            normalMapTexture = loadTextureFromFile(texturePath.c_str(), "");
        }
        else {
            // Assign the default flat normal map
            normalMapTexture = flatNormalMap;
        }

        // Error handling if normal map fails to load
        if (normalMapTexture == 0) {
            std::cerr << "Failed to load normal map for material: " << matName << std::endl;
        }

        meshes.push_back(Mesh(vertices, indices, diffuseTexture, normalMapTexture));
    }

    return meshes;
}

GLuint loadTextureFromFile(const char* path, const std::string&) {
    GLuint textureID;
    glGenTextures(1, &textureID);

    int width, height, nrComponents;
    // Force stb_image to load 4 components (RGBA)
    unsigned char* data = stbi_load(path, &width, &height, &nrComponents, 4);
    if (data) {
        GLenum format = GL_RGBA;

        glBindTexture(GL_TEXTURE_2D, textureID);
        // Use GL_RGBA for both internal format and format
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, format, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);

        // Set texture parameters
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        stbi_image_free(data);
    }
    else {
        std::cerr << "Texture failed to load at path: " << path << std::endl;
        stbi_image_free(data);
    }

    return textureID;
}

void attachSharedDepthBuffer(GLuint framebuffer, GLuint depthMap) {
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, depthMap, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0); // Unbind framebuffer after attaching depth
}

GLuint createFramebuffer(GLuint& colorTexture, GLenum internalFormat, GLenum format, GLenum type, int width, int height) {
    GLuint framebuffer;
    glGenFramebuffers(1, &framebuffer);
    glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);

    glGenTextures(1, &colorTexture);
    glBindTexture(GL_TEXTURE_2D, colorTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, colorTexture, 0);

    GLenum attachments[1] = { GL_COLOR_ATTACHMENT0 };
    glDrawBuffers(1, attachments);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
        std::cerr << "Framebuffer not complete!" << std::endl;
    }

    glBindFramebuffer(GL_FRAMEBUFFER, 0); // Unbind framebuffer after setup
    return framebuffer;
}

struct Framebuffer {
    GLuint framebuffer;
    GLuint colorTexture;

    Framebuffer(GLenum internalFormat, GLenum format, GLenum type, int width, int height) {
        framebuffer = createFramebuffer(colorTexture, internalFormat, format, type, width, height);
    }

    void attachDepthBuffer(GLuint depthMap) {
        attachSharedDepthBuffer(framebuffer, depthMap);
    }
};


int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Create a GLFW window
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3); // Request OpenGL 4.3 or newer
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(WIDTH, HEIGHT, "OpenGL Basic Application", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable VSync to cap frame rate to monitor's refresh rate
    glfwSetCursorPosCallback(window, mouseCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    glfwSetScrollCallback(window, scrollCallback);

    // Initialize GLEW
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    // Clear any GLEW errors
    glGetError(); // Clear error flag set by GLEW

    // Enable OpenGL debugging if supported
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(MessageCallback, nullptr);

    // Optionally filter which types of messages you want to log
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);

    // Define the viewport dimensions
    glViewport(0, 0, WIDTH, HEIGHT);

    glEnable(GL_DEPTH_TEST);

    glCullFace(GL_BACK); // Cull back faces (default)

    // Load the shared lightmap texture
    std::string lightmapPath = FileSystemUtils::getAssetFilePath("textures/tutorialLightingMap.tga");
    lightmapTexture = loadTextureFromFile(lightmapPath.c_str(), "");

    // Load the model
    std::vector<Mesh> meshes = loadModel(FileSystemUtils::getAssetFilePath("models/tutorial_map.fbx"));

    // Build and compile the shader program
   // Vertex Shader
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
    glCompileShader(vertexShader);

    // Check for shader compile errors
    int success;
    char infoLog[512];
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cerr << "ERROR::VERTEX_SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // Fragment Shader
    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
    glCompileShader(fragmentShader);

    // Check for shader compile errors
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cerr << "ERROR::FRAGMENT_SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // Link shaders
    GLuint shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    // Check for linking errors
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cerr << "ERROR::SHADER_PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);

    // Quad shaders
    GLuint quadVertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(quadVertexShader, 1, &quadVertexShaderSource, NULL);
    glCompileShader(quadVertexShader);

    glGetShaderiv(quadVertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(quadVertexShader, 512, NULL, infoLog);
        std::cerr << "ERROR::QUAD_VERTEX_SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    GLuint quadShaderProgram = glCreateProgram();
    glAttachShader(quadShaderProgram, quadVertexShader);
    glLinkProgram(quadShaderProgram);

    glGetProgramiv(quadShaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(quadShaderProgram, 512, NULL, infoLog);
        std::cerr << "ERROR::QUAD_SHADER_PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }

    glDeleteShader(quadVertexShader);

    // New Vertex Shader
    GLuint newVertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(newVertexShader, 1, &newVertexShaderSource, NULL);
    glCompileShader(newVertexShader);

    // Check for shader compile errors
    glGetShaderiv(newVertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(newVertexShader, 512, NULL, infoLog);
        std::cerr << "ERROR::NEW_VERTEX_SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // New Fragment Shader
    GLuint newFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(newFragmentShader, 1, &newFragmentShaderSource, NULL);
    glCompileShader(newFragmentShader);

    // Check for shader compile errors
    glGetShaderiv(newFragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        glGetShaderInfoLog(newFragmentShader, 512, NULL, infoLog);
        std::cerr << "ERROR::NEW_FRAGMENT_SHADER::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    // Link shaders
    GLuint newShaderProgram = glCreateProgram();
    glAttachShader(newShaderProgram, newVertexShader);
    glAttachShader(newShaderProgram, newFragmentShader);
    glLinkProgram(newShaderProgram);

    // Check for linking errors
    glGetProgramiv(newShaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        glGetProgramInfoLog(newShaderProgram, 512, NULL, infoLog);
        std::cerr << "ERROR::NEW_SHADER_PROGRAM::LINKING_FAILED\n" << infoLog << std::endl;
    }

    glDeleteShader(newVertexShader);
    glDeleteShader(newFragmentShader);

    // Combined Quad shaders
    GLuint combinedQuadVertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(combinedQuadVertexShader, 1, &quadVertexShaderSource, NULL);
    glCompileShader(combinedQuadVertexShader);
    // Check for compilation errors...

    GLuint combinedQuadFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(combinedQuadFragmentShader, 1, &combinedQuadFragmentShaderSource, NULL);
    glCompileShader(combinedQuadFragmentShader);
    // Check for compilation errors...

    GLuint combinedQuadShaderProgram = glCreateProgram();
    glAttachShader(combinedQuadShaderProgram, combinedQuadVertexShader);
    glAttachShader(combinedQuadShaderProgram, combinedQuadFragmentShader);
    glLinkProgram(combinedQuadShaderProgram);
    // Check for linking errors...

    // Delete shaders after linking
    glDeleteShader(combinedQuadVertexShader);
    glDeleteShader(combinedQuadFragmentShader);

    /* FBO setup */

    // Create depth texture
    GLuint depthMap;
    glGenTextures(1, &depthMap);
    glBindTexture(GL_TEXTURE_2D, depthMap);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT32, WIDTH, HEIGHT, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

    // Lightmap Pass
    Framebuffer lightmapPass(GL_RGB8, GL_RGB, GL_UNSIGNED_BYTE, WIDTH, HEIGHT);
    lightmapPass.attachDepthBuffer(depthMap);

    // Normal-specular Pass
    Framebuffer normalMapPass(GL_RGB8, GL_RGB, GL_UNSIGNED_BYTE, WIDTH, HEIGHT);
    normalMapPass.attachDepthBuffer(depthMap);

    // Set up the quad VAO
    GLuint quadVAO, quadVBO;
    float quadVertices[] = {
        // positions   // texCoords
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

    int debugMode = 0;  // Initialize outside the render loop
    bool keyPressed = false;  // Track key press state

    // Set light direction
    glm::vec3 lightDir = glm::normalize(glm::vec3(0.5f, -1.0f, 0.3f));

    // Render loop
    while (!glfwWindowShouldClose(window)) {
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        processInput(window);

        // Input handling
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            glfwSetWindowShouldClose(window, true);

        // Handle key inputs to toggle between modes
        if (glfwGetKey(window, GLFW_KEY_1) == GLFW_PRESS) {
            debugMode = 0;  // Show combined lightmap pass
        }
        if (glfwGetKey(window, GLFW_KEY_2) == GLFW_PRESS) {
            debugMode = 1;  // Show diffuse texture only
        }
        if (glfwGetKey(window, GLFW_KEY_3) == GLFW_PRESS) {
            debugMode = 2;  // Show lightmap texture only
        }
        if (glfwGetKey(window, GLFW_KEY_4) == GLFW_PRESS) {
            debugMode = 3;  // Show normal mapping and specular lighting pass
        }
        if (glfwGetKey(window, GLFW_KEY_5) == GLFW_PRESS) {
            debugMode = 4;  // Show normal map only
        }
        if (glfwGetKey(window, GLFW_KEY_6) == GLFW_PRESS) {
            debugMode = 5;  // Show specular component only
        }

        // ========== First Pass: Lightmap Rendering ==========
        glBindFramebuffer(GL_FRAMEBUFFER, lightmapPass.framebuffer);
        glClearColor(0.3f, 0.3f, 0.4f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Use the shader program for lightmapping
        glUseProgram(shaderProgram);

        // Set the 'debugMode' uniform
        glUniform1i(glGetUniformLocation(shaderProgram, "debugMode"), debugMode);

        // Set up view and projection matrices
        glm::mat4 view = camera.getViewMatrix();
        glm::mat4 projection = camera.getProjectionMatrix((float)WIDTH / (float)HEIGHT);

        // Pass view and projection matrices to the shader
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));

        // Set the blendFactor to control the lightmap influence
        glUniform1f(glGetUniformLocation(shaderProgram, "blendFactor"), 1.0f);

        // Bind lightmap texture
        glActiveTexture(GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, lightmapTexture);
        glUniform1i(glGetUniformLocation(shaderProgram, "lightmapTexture"), 2);

        // Render all meshes
        for (const auto& mesh : meshes) {
            glm::mat4 model = glm::mat4(1.0f);
            model = glm::scale(model, glm::vec3(0.01f));
            model = glm::rotate(model, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
            glUniformMatrix4fv(glGetUniformLocation(shaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));
            mesh.Draw(shaderProgram);
        }

        // Unbind the framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // ========== Second Pass: Normal Mapping and Specular Lighting ==========
        // Second Pass: Normal Mapping and Specular Lighting
        glBindFramebuffer(GL_FRAMEBUFFER, normalMapPass.framebuffer);
        glClearColor(0.3f, 0.3f, 0.4f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Use the shader program for normal mapping
        glUseProgram(newShaderProgram);

        // Set the 'debugMode' uniform
        glUniform1i(glGetUniformLocation(newShaderProgram, "debugMode"), debugMode);

        // Pass view and projection matrices to the shader
        glUniformMatrix4fv(glGetUniformLocation(newShaderProgram, "view"), 1, GL_FALSE, glm::value_ptr(view));
        glUniformMatrix4fv(glGetUniformLocation(newShaderProgram, "projection"), 1, GL_FALSE, glm::value_ptr(projection));
        glUniform3fv(glGetUniformLocation(newShaderProgram, "lightDir"), 1, glm::value_ptr(lightDir));
        glUniform3fv(glGetUniformLocation(newShaderProgram, "viewPos"), 1, glm::value_ptr(camera.getPosition()));

        // Render all meshes
        for (const auto& mesh : meshes) {
            glm::mat4 model = glm::mat4(1.0f);
            model = glm::scale(model, glm::vec3(0.01f));
            model = glm::rotate(model, glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
            glUniformMatrix4fv(glGetUniformLocation(newShaderProgram, "model"), 1, GL_FALSE, glm::value_ptr(model));

            // Bind diffuse texture (for specular mask)
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, mesh.diffuseTexture);
            glUniform1i(glGetUniformLocation(newShaderProgram, "diffuseTexture"), 0);

            // Bind normal map texture
            glActiveTexture(GL_TEXTURE1);
            glBindTexture(GL_TEXTURE_2D, mesh.normalMapTexture);
            glUniform1i(glGetUniformLocation(newShaderProgram, "normalMap"), 1);

            // Draw mesh
            mesh.DrawWithoutDiffuse(newShaderProgram);
        }

        // Unbind the framebuffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        // ========== Final Pass: Combine Textures ==========
        glClear(GL_COLOR_BUFFER_BIT);
        glDisable(GL_DEPTH_TEST);

        // Use the combined quad shader program
        glUseProgram(combinedQuadShaderProgram);

        // Bind the first pass texture
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, lightmapPass.colorTexture);
        glUniform1i(glGetUniformLocation(combinedQuadShaderProgram, "firstPassTexture"), 0);

        // Bind the second pass texture
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, normalMapPass.colorTexture);
        glUniform1i(glGetUniformLocation(combinedQuadShaderProgram, "secondPassTexture"), 1);

        // Render the quad
        glBindVertexArray(quadVAO);
        glDrawArrays(GL_TRIANGLES, 0, 6);

        // Re-enable depth testing
        glEnable(GL_DEPTH_TEST);

        // Swap buffers and poll IO events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Clean up
    glDeleteProgram(shaderProgram);
    glDeleteProgram(quadShaderProgram);
    glDeleteProgram(newShaderProgram);
    glDeleteVertexArrays(1, &quadVAO);
    glDeleteBuffers(1, &quadVBO);
    glDeleteFramebuffers(1, &depthMap);

    glfwTerminate();
    return 0;
}