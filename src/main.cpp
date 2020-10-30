
#include <windef.h>
#include <glad/glad.h>  
#include <GL/glu.h>
#include <GLFW/glfw3.h>
#include <iostream>
#include <vector>
#include "Eigen/Dense"
#include <Eigen/Cholesky>  
#include <Eigen/LU>  
#include <Eigen/QR>  
#include <Eigen/SVD>  

#include <iostream>
#include "../3rdLibs/glm/glm/glm.hpp"
#include "../3rdLibs/glm/glm/gtc/matrix_transform.hpp"
#include "../3rdLibs/glm/glm/gtc/type_ptr.hpp"
#include "../3rdLibs/imgui/imgui.h"
#include "../3rdLibs/imgui/imgui_impl_glfw.h"
#include "../3rdLibs/imgui/imgui_impl_opengl3.h"
#include "../3rdLibs/imgui/imconfig.h"
#include "../3rdLibs/imgui/imgui_internal.h"
#include "../3rdLibs/imgui/imstb_rectpack.h"
#include "../3rdLibs/imgui/imstb_textedit.h"
#include "../3rdLibs/imgui/imstb_truetype.h"

#include "../inc/my_texture.h"
#include "../inc/shader_m.h"
#include "tiny_obj_loader.h"

#include <time.h>


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// The .obj .mtl and images are in Dir "model".                                                                  //
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*-----------------------------------------------------------------------*/
//Here are some mouse and keyboard function. You can change that.
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);

const unsigned int SCR_WIDTH = 1200;
const unsigned int SCR_HEIGHT = 900;
float lastX = SCR_WIDTH / 2.0f;
float lastY = SCR_HEIGHT / 2.0f;
float degreeX = (360 * lastX / 400);
float degreeY = (360 * lastY / 300);
bool firstMouse = true;
float deltaTime = 0.0f;	// time between current frame and last frame
float lastFrame = 0.0f;
float OX = 0;//should be update to a new coordinate
float OY = 0;
float OZ = 0;
// camera parameter
glm::vec3 cameraPos = glm::vec3(0.0f, 0.0f, 3.0f);
glm::vec3 cameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
float currentFrame;
float yaw = -90.0;
float pitch = 0;
// yaw and pitch for model
float model_yaw = 0.0f;
float model_pitch = 0.0f;
// parameters for b-spine curve (initial default)
bool playSwitch = false;
float animTime = 5.0f;
int curveP = 2; // deg-2
int curveN = -1; // total n+1 data points
int curveM = curveN + curveP + 1; // total m+1 knots
std::vector<float> curveTSet;
std::vector<Eigen::Vector2f> curveDSet;
std::vector<float> curveUSet;
std::vector<Eigen::Vector2f> curvePSet;

// new functions here
void restart() {
	curveN = -1;
	curveM = curveN + curveP + 1;
	model_yaw = 0.0f;
	model_pitch = 0.0f;
	curveTSet.clear();
	curveDSet.clear();
	curveUSet.clear();
	curvePSet.clear();
}


void addD(Eigen::Vector2f dataP, float t) {
	curveDSet.push_back(dataP);
	curveTSet.push_back(t);
	curveN += 1;
	curveM = curveN + curveP + 1;
}


void setSample() {
	animTime = 5.0f;
	curveP = 2; // deg-2
	curveM = curveN + curveP + 1; // total m+1 knots
	// test for curve
	addD(Eigen::Vector2f(0.0f, 0.0f), 0.0f);
	addD(Eigen::Vector2f(45.0f, 20.0f), 0.1f);
	addD(Eigen::Vector2f(90.0f, 0.0f), 0.25f);
	addD(Eigen::Vector2f(45.0f, -20.0f), 0.4f);
	addD(Eigen::Vector2f(0.0f, 0.0f), 0.5f);
	addD(Eigen::Vector2f(-45.0f, 20.0f), 0.6f);
	addD(Eigen::Vector2f(-90.0f, 0.0f), 0.75f);
	addD(Eigen::Vector2f(-45.0f, -20.0f), 0.9f);
	addD(Eigen::Vector2f(0.0f, 0.0f), 1.0f);
}

std::vector<float> getNC(float u) {
	std::vector<float> ret;
	// initialization
	for (int i = 0; i <= curveN; i++) {
		ret.push_back(0.0f);
	}
	if (u == curveUSet[0]) {
		ret[0] = 1.0f;
		return ret;
	}
	else if (u == curveUSet[curveM]) {
		ret[curveN] = 1.0f;
		return ret;
	}
	// find where the k is
	int k = 0;
	for (int i = 0; i <= curveM - 1; i++) {
		if (curveUSet[i] <= u && u < curveUSet[i + 1]) {
			k = i;
			break;
		}
	}
	ret[k] = 1.0f;
	for (int d = 1; d <= curveP; d++) {
		ret[k - d] = (curveUSet[k + 1] - u) / (curveUSet[k + 1] - curveUSet[(k - d) + 1]) * ret[(k - d) + 1];
		for (int i = k - d + 1; i <= k - 1; i++) {
			ret[i] = (u - curveUSet[i]) / (curveUSet[i + d] - curveUSet[i]) * ret[i] + (curveUSet[i + d + 1] - u) / (curveUSet[i + d + 1] - curveUSet[i + 1]) * ret[i + 1];
		}
		ret[k] = (u - curveUSet[k]) / (curveUSet[k + d] - curveUSet[k]) * ret[k];
	}
	return ret;
}

void getP() {
	// first build Uset
	int restU = curveM + 1 - 2 * (curveP + 1);
	float Upace = 1.0f / float(restU + 1);
	for (int i = 0; i <= curveM; i++) {
		if (i >= 0 && i <= curveP) {
			curveUSet.push_back(0.0f);
		}
		else if (i >= curveM - curveP && i <= curveM) {
			curveUSet.push_back(1.0f);
		}
		else {
			curveUSet.push_back(Upace * (i - curveP));
		}
		//printf("%d: %f\n", i, curveUSet[i]);
	}
	// Matrix N
	const int temp = curveN + 1;
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> matN(temp, temp);
	//matN(2, 3) = 1.0f;
	//printf("rows: %d, columns: %d\n", matN.rows(), matN.cols());
	//printf("%f %f\n", matN(2, 3), matN(0, 0));
	/*std::vector<float> test1 = getNC(0.0f);
	std::vector<float> test2 = getNC(0.25f);
	for (int i = 0; i < test1.size(); i++) {
		printf("%f ", test1[i]);
	}
	printf("\n");
	for (int i = 0; i < test2.size(); i++) {
		printf("%f ", test2[i]);
	}*/
	for (int i = 0; i <= curveN; i++) {
		std::vector<float> NC = getNC(curveTSet[i]);
		for (int j = 0; j <= curveN; j++) {
			matN(i, j) = NC[j];
		}
	}
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> matD(temp, 2);
	for (int i = 0; i <= curveN; i++) {
		matD(i, 0) = curveDSet[i][0];
		matD(i, 1) = curveDSet[i][1];
		//printf("%f, %f", matD(i, 0), matD(i, 1));
	}
	Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> sol(temp, 2);
	sol = matN.lu().solve(matD);
	// put sol to PSet
	for (int i = 0; i <= curveN; i++) {
		//printf("%f, %f\n", sol(i, 0), sol(i, 1));
		curvePSet.push_back(Eigen::Vector2f(sol(i, 0), sol(i, 1)));
	}
}


void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
	if (firstMouse)
	{
		lastX = xpos;
		lastY = ypos;
		firstMouse = false;
	}
	float xoffset = xpos - lastX;
	float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top
	lastX = xpos;
	lastY = ypos;
	float sensitivity = 0.1;
	xoffset *= sensitivity;
	yoffset *= sensitivity;
	yaw += xoffset;
	pitch += yoffset;
	if (pitch > 89.0f)
		pitch = 89.0f;
	if (pitch < -89.0f)
		pitch = -89.0f;
	glm::vec3 front;//why not in global 
	front.x = cos(glm::radians(pitch)) * cos(glm::radians(yaw));
	front.y = sin(glm::radians(pitch));
	front.z = cos(glm::radians(pitch)) * sin(glm::radians(yaw));
	cameraFront = glm::normalize(front);
	//std::cout << yaw << " " << pitch << std::endl;
}

void processInput(GLFWwindow* window)
{
	/*currentFrame = glfwGetTime();
	deltaTime = currentFrame - lastFrame;
	lastFrame = currentFrame;*/
	float cameraSpeed = 2.0f * deltaTime; // adjust accordingly
	float modelYawSpeed = 200.0f * deltaTime;
	float modelPitchSpeed = 200.0f * deltaTime;
	/*if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		cameraPos += cameraSpeed * cameraFront;
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		cameraPos -= cameraSpeed * cameraFront;
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		cameraPos -= glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		cameraPos += glm::normalize(glm::cross(cameraFront, cameraUp)) * cameraSpeed;*/
	if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
		glfwSetWindowShouldClose(window, true);
	if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
		model_pitch += modelPitchSpeed;
	if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
		model_pitch -= modelPitchSpeed;
	if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
		model_yaw += modelYawSpeed;
	if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
		model_yaw -= modelYawSpeed;
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
	glViewport(0, 0, width, height);
}
// no use
void initPMV()
{
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();

	gluPerspective(60, SCR_WIDTH / SCR_HEIGHT, 0.1, 1000);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt
	(
		3, 3, 3,
		0, 0, 0,
		0, 1, 0
	);

}
// no use
void changePMV()
{


}
/*-----------------------------------------------------------------------*/



//This an function to get v, vt and vn. 
bool make_face(std::vector<float> v, std::vector<float> vt, std::vector<float> vn, std::vector<unsigned int> f,
	std::vector<glm::vec3>& points, std::vector<glm::vec3>& normals, std::vector<glm::vec2>& uvs)
{
	if (f.size() % 3 != 0)
		return false;
	// i trangles
	for (int i = 0; i < f.size()/3; i += 1)
	{
		int k = i * 3;
		// three vertexs of triangle
		for (int j = 0; j < 3; j++)
		{
			points.push_back(glm::vec3(v[f[k + j] * 3], v[f[k + j] * 3 + 1], v[f[k + j] * 3 + 2]));
			normals.push_back(glm::vec3(vn[f[k + j] * 3], vn[f[k + j] * 3 + 1], vn[f[k + j] * 3 + 2]));
			uvs.push_back(glm::vec2(vt[f[k + j] * 2], vt[f[k + j] * 2 + 1]));
		}
		
	}
}
// no use
void get_vec3(std::vector<float> list, std::vector<glm::vec3> &vec)
{
	int n = list.size() / 3;
	for (int i = 0; i < n; i++)
	{
		vec.push_back(glm::vec3(list[i], list[i + 1], list[i + 2]));
	}
}
// no use
void get_vec2(std::vector<float> list, std::vector<glm::vec2>& vec)
{
	int n = list.size() / 2;
	for (int i = 0; i < n; i++)
	{
		vec.push_back(glm::vec2(list[i], list[i + 1]));
	}
}



int main()
{
	// test for curve
	/*addD(Eigen::Vector2f(0.0f, 0.0f), 0.0f);
	addD(Eigen::Vector2f(45.0f, 20.0f), 0.1f);
	addD(Eigen::Vector2f(90.0f, 0.0f), 0.25f);
	addD(Eigen::Vector2f(45.0f, -20.0f), 0.4f);
	addD(Eigen::Vector2f(0.0f, 0.0f), 0.5f);
	addD(Eigen::Vector2f(-45.0f, 20.0f), 0.6f);
	addD(Eigen::Vector2f(-90.0f, 0.0f), 0.75f);
	addD(Eigen::Vector2f(-45.0f, -20.0f), 0.9f);
	addD(Eigen::Vector2f(0.0f, 0.0f), 1.0f);
	getP();*/
    // glfw: initialize and configure
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // uncomment this statement to fix compilation on OS X
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	//glfwSetCursorPosCallback(window, mouse_callback);
	
    gladLoadGL();  
	// hide the cursor
	//glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

	glEnable(GL_DEPTH_TEST);

	// build gui //
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO(); (void)io;
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
	//io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
	// Setup Dear ImGui style
	ImGui::StyleColorsDark();
	//ImGui::StyleColorsClassic();
	// Setup Platform/Renderer bindings
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 130");

	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Here you need to fill construct function of class Shader. And you need to understand other funtions in Shader.//
	// Then, write code in shader_m.vs, shader_m.fs and shader_m.gs to finish the tasks.                             //
	///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	Shader my_shader(
		"../src/shader_m.vs", 
		"../src/shader_m.fs"
	);
	//A shader for light visiable source
	Shader lampShader("../src/lamp.vs", "../src/lamp.fs");
	


	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// tiny::LoadObj is a function to load obj file. The output is shape_t and material_t.                         //
	// "shape.mesh" is a mesh struct. "mesh.positions", "mesh.normals", "mesh.texcoords" corresponds to v,vn,vt.   //
	// For example:                                                                                                //
	// positions[0],positions[1],positions[2] -> v 0,0,1                                                           //
	// positions[3],positions[4],positions[5] -> v 0,1,0                                                           //
	// "mesh.indice" corresponds to f, but it is different from f. Each element is an index for all of v,vn,vt.    //
	// positions[0],positions[1],positions[2] -> v 0,0,1  positions[0],positions[1],positions[2] -> v 0,0,1        //
	// You can read tiny_obj_loader.h to get more specific information.                                            //
	//                                                                                                             //
	// I have write make_face for you.  It will return v, vt, vn in vec form (each element if for one point).      //
	// These informations can help you to do normal mapping.  (You can calculate tangent here)                     //
	// Since i did not assign uv for noraml map, you just need use vt as uv for normal map, but you will find it is//
	//  ugly. So please render a box to show a nice normal mapping. (Do normal mapping on obj and box)             //
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// how many shapes
	std::vector<tinyobj::shape_t> shapes;
	// no use
	std::vector<tinyobj::material_t> materials;
	std::string if_load_succeed = tinyobj::LoadObj(shapes, materials,
		"../thinker-obj/pikaqiu.obj"
	);
	//printf("%s\n", if_load_succeed);
	// three list for infomation of each shape
	std::vector<unsigned int> obj_VBO_l, obj_VAO_l, obj_NUM_l;
	// temp
	unsigned int tVBO, tVAO;
	// calculation helper
	glm::vec3 edge1, edge2, tangent, bitangent;
	glm::vec2 deltaUV1, deltaUV2;
	float f;
	// fixed data for exception
	glm::vec3 bakTangent(1, 0, 0);
	glm::vec3 bakBitangent(0, 1, 0);
	glm::vec3 lastT, lastB;
	for (int i = 0; i < shapes.size(); i++)
	{
		
		std::vector < glm::vec3 > out_vertices;
		std::vector < glm::vec2 > out_uvs;
		std::vector < glm::vec3 > out_normals;

		// out_vertices, out_uvs, out_normals will get v, vt and vn.
		make_face(shapes[i].mesh.positions, shapes[i].mesh.texcoords, shapes[i].mesh.normals, shapes[i].mesh.indices,
			out_vertices, out_normals, out_uvs);
		unsigned int tVBO, tVAO;
		// temp tVertices able to change if i > 1? Yes
		std::vector<float> tVertices;
		// all vertices of one shape
		for (int j = 0; j < out_vertices.size(); j++) {
			// pos
			tVertices.push_back(out_vertices[j].x); tVertices.push_back(out_vertices[j].y); tVertices.push_back(out_vertices[j].z);
			// normal
			tVertices.push_back(out_normals[j].x); tVertices.push_back(out_normals[j].y); tVertices.push_back(out_normals[j].z);
			// uvs
			tVertices.push_back(out_uvs[j].x); tVertices.push_back(out_uvs[j].y);
			// T B
			// one triangle one calculation
			if (j % 3 == 0) {
				edge1 = out_vertices[j + 1] - out_vertices[j];
				edge2 = out_vertices[j + 2] - out_vertices[j];
				deltaUV1 = out_uvs[j + 1] - out_uvs[j];
				deltaUV2 = out_uvs[j + 2] - out_uvs[j];
				// exception
				if (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y == 0) {
					// T
					tVertices.push_back(bakTangent.x); tVertices.push_back(bakTangent.y); tVertices.push_back(bakTangent.z);
					// B
					tVertices.push_back(bakBitangent.x); tVertices.push_back(bakBitangent.y); tVertices.push_back(bakBitangent.z);
					// record
					lastT = bakTangent; lastB = bakBitangent;
				}
				// do calculation 
				else {
					f = 1 / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);
					tangent.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
					tangent.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
					tangent.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);
					tangent = glm::normalize(tangent);
					bitangent.x = f * (-deltaUV2.x * edge1.x + deltaUV1.x * edge2.x);
					bitangent.y = f * (-deltaUV2.x * edge1.y + deltaUV1.x * edge2.y);
					bitangent.z = f * (-deltaUV2.x * edge1.z + deltaUV1.x * edge2.z);
					bitangent = glm::normalize(bitangent);
					// T
					tVertices.push_back(tangent.x); tVertices.push_back(tangent.y); tVertices.push_back(tangent.z);
					// B
					tVertices.push_back(bitangent.x); tVertices.push_back(bitangent.y); tVertices.push_back(bitangent.z);
					lastT = tangent; lastB = bitangent; // record
				}
			}
			// the other two vertices
			else {
				tVertices.push_back(lastT.x); tVertices.push_back(lastT.y); tVertices.push_back(lastT.z);
				tVertices.push_back(lastB.x); tVertices.push_back(lastB.y); tVertices.push_back(lastB.z);
			}
		}
		// set attributes for tVAO tVBO
		glGenVertexArrays(1, &tVAO);
		glGenBuffers(1, &tVBO);
		glBindVertexArray(tVAO);
		glBindBuffer(GL_ARRAY_BUFFER, tVBO);
		glBufferData(GL_ARRAY_BUFFER, tVertices.size()*sizeof(float), &tVertices[0], GL_STATIC_DRAW);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 14 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0); // pos
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 14 * sizeof(float), (void*)(3 * sizeof(float)));
		glEnableVertexAttribArray(1); // normal
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 14 * sizeof(float), (void*)(6 * sizeof(float)));
		glEnableVertexAttribArray(2); // uvs
		glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 14 * sizeof(float), (void*)(8 * sizeof(float)));
		glEnableVertexAttribArray(3); // T
		glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 14 * sizeof(float), (void*)(11 * sizeof(float)));
		glEnableVertexAttribArray(4); // B
		// push to VAO,VBO,NUM list
		obj_VBO_l.push_back(tVBO); obj_VAO_l.push_back(tVAO); obj_NUM_l.push_back(out_vertices.size());
	}
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Render a box to show nice normal mapping.                                                                   //
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	float vertices_cube_0[] = {

		// positions          // normals           // texture coords

		-0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f,
		 0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f,  0.0f,
		 0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f,  1.0f,
		 0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  1.0f,  1.0f,
		-0.5f,  0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f,  1.0f,
		-0.5f, -0.5f, -0.5f,  0.0f,  0.0f, -1.0f,  0.0f,  0.0f,

		-0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,
		 0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f,  0.0f,
		 0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f,  1.0f,
		 0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  1.0f,  1.0f,
		-0.5f,  0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f,  1.0f,
		-0.5f, -0.5f,  0.5f,  0.0f,  0.0f,  1.0f,  0.0f,  0.0f,

		-0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f,  0.0f,
		-0.5f,  0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  1.0f,  1.0f,
		-0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f,  1.0f,
		-0.5f, -0.5f, -0.5f, -1.0f,  0.0f,  0.0f,  0.0f,  1.0f,
		-0.5f, -0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  0.0f,  0.0f,
		-0.5f,  0.5f,  0.5f, -1.0f,  0.0f,  0.0f,  1.0f,  0.0f,

		 0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,
		 0.5f,  0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  1.0f,  1.0f,
		 0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  0.0f,  1.0f,
		 0.5f, -0.5f, -0.5f,  1.0f,  0.0f,  0.0f,  0.0f,  1.0f,
		 0.5f, -0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  0.0f,  0.0f,
		 0.5f,  0.5f,  0.5f,  1.0f,  0.0f,  0.0f,  1.0f,  0.0f,

		-0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f,  1.0f,
		 0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  1.0f,  1.0f,
		 0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f,  0.0f,
		 0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  1.0f,  0.0f,
		-0.5f, -0.5f,  0.5f,  0.0f, -1.0f,  0.0f,  0.0f,  0.0f,
		-0.5f, -0.5f, -0.5f,  0.0f, -1.0f,  0.0f,  0.0f,  1.0f,

		-0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f,
		 0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  1.0f,  1.0f,
		 0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f,  0.0f,
		 0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  1.0f,  0.0f,
		-0.5f,  0.5f,  0.5f,  0.0f,  1.0f,  0.0f,  0.0f,  0.0f,
		-0.5f,  0.5f, -0.5f,  0.0f,  1.0f,  0.0f,  0.0f,  1.0f

	};
	// operation on cube vertices
	std::vector<glm::vec3> cube_vertices, cube_normals;
	std::vector<glm::vec2> cube_uvs;
	std::vector<float> vertices_cube_1;
	// reload cube data
	for (int j = 0; j < 36; j++) {
		cube_vertices.push_back(glm::vec3(vertices_cube_0[8 * j], vertices_cube_0[8 * j + 1], vertices_cube_0[8 * j + 2]));
		cube_normals.push_back(glm::vec3(vertices_cube_0[8 * j + 3], vertices_cube_0[8 * j + 4], vertices_cube_0[8 * j + 5]));
		cube_uvs.push_back(glm::vec2(vertices_cube_0[8 * j + 6], vertices_cube_0[8 * j + 7]));
	}
	// calculate TB, 36 vertices
	for (int j = 0; j < 36; j++) {
		vertices_cube_1.push_back(cube_vertices[j].x); vertices_cube_1.push_back(cube_vertices[j].y); vertices_cube_1.push_back(cube_vertices[j].z);
		vertices_cube_1.push_back(cube_normals[j].x); vertices_cube_1.push_back(cube_normals[j].y); vertices_cube_1.push_back(cube_normals[j].z);
		vertices_cube_1.push_back(cube_uvs[j].x); vertices_cube_1.push_back(cube_uvs[j].y);
		// each triangle
		if (j % 3 == 0) {
			edge1 = cube_vertices[j + 1] - cube_vertices[j];
			edge2 = cube_vertices[j + 2] - cube_vertices[j];
			deltaUV1 = cube_uvs[j + 1] - cube_uvs[j];
			deltaUV2 = cube_uvs[j + 2] - cube_uvs[j];
			if (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y == 0) {
				vertices_cube_1.push_back(bakTangent.x); vertices_cube_1.push_back(bakTangent.y); vertices_cube_1.push_back(bakTangent.z);
				vertices_cube_1.push_back(bakBitangent.x); vertices_cube_1.push_back(bakBitangent.y); vertices_cube_1.push_back(bakBitangent.z);
				lastT = bakTangent; lastB = bakBitangent;
			}
			else {
				f = 1 / (deltaUV1.x * deltaUV2.y - deltaUV2.x * deltaUV1.y);
				tangent.x = f * (deltaUV2.y * edge1.x - deltaUV1.y * edge2.x);
				tangent.y = f * (deltaUV2.y * edge1.y - deltaUV1.y * edge2.y);
				tangent.z = f * (deltaUV2.y * edge1.z - deltaUV1.y * edge2.z);
				tangent = glm::normalize(tangent);
				bitangent.x = f * (-deltaUV2.x * edge1.x + deltaUV1.x * edge2.x);
				bitangent.y = f * (-deltaUV2.x * edge1.y + deltaUV1.x * edge2.y);
				bitangent.z = f * (-deltaUV2.x * edge1.z + deltaUV1.x * edge2.z);
				bitangent = glm::normalize(bitangent);
				vertices_cube_1.push_back(tangent.x); vertices_cube_1.push_back(tangent.y); vertices_cube_1.push_back(tangent.z);
				vertices_cube_1.push_back(bitangent.x); vertices_cube_1.push_back(bitangent.y); vertices_cube_1.push_back(bitangent.z);
				lastT = tangent; lastB = bitangent;
			}
		}
		else {
			vertices_cube_1.push_back(lastT.x); vertices_cube_1.push_back(lastT.y); vertices_cube_1.push_back(lastT.z);
			vertices_cube_1.push_back(lastB.x); vertices_cube_1.push_back(lastB.y); vertices_cube_1.push_back(lastB.z);
		}
	}

	unsigned int cubeVBO, cubeVAO;
	glGenVertexArrays(1, &cubeVAO);
	glGenBuffers(1, &cubeVBO);
	// bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
	glBindVertexArray(cubeVAO);

	glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
	glBufferData(GL_ARRAY_BUFFER, vertices_cube_1.size() * sizeof(float), &vertices_cube_1[0], GL_STATIC_DRAW);

	// position attribute
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 14 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	// normal attribute
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 14 * sizeof(float), (void*)(3 * sizeof(float)));
	glEnableVertexAttribArray(1);
	// texture coords
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 14 * sizeof(float), (void*)(6 * sizeof(float)));
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(3, 3, GL_FLOAT, GL_FALSE, 14 * sizeof(float), (void*)(8 * sizeof(float)));
	glEnableVertexAttribArray(3);
	glVertexAttribPointer(4, 3, GL_FLOAT, GL_FALSE, 14 * sizeof(float), (void*)(11 * sizeof(float)));
	glEnableVertexAttribArray(4);
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// You need to fill this function which is defined in my_texture.h. The parameter is the path of your image.   //
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// cat face
	unsigned int texture1 = loadTexture("../thinker-obj/Textures/p_r.jpg");
	unsigned int texture2 = loadTexture("../thinker-obj/Textures/eye.jpg");

	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	// Here we defined pointlights in shader and passed some parameter for you. You can take this as an example.   //
	// Or you can change it if you like.                                                                           //
	/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	glm::vec3 pointLightPositions[] = {
		glm::vec3(5.7f,  5.2f,  0.0f),
		glm::vec3(2.3f, -3.3f, 0.0f),
		glm::vec3(-4.0f,  2.0f, 0.0f),
		glm::vec3(0.0f,  0.0f, 8.0f)
	};
	my_shader.use();
	my_shader.setVec3("dirLight.direction", glm::vec3(1.01f, 1.01f, 1.01f));
	my_shader.setVec3("dirLightDir2VS", glm::vec3(1.01f, 1.01f, 1.01f));
	my_shader.setVec3("dirLight.ambient", glm::vec3(0.01f, 0.01f, 0.02f));
	my_shader.setVec3("dirLight.diffuse", glm::vec3(1.0f, 1.0f, 1.0f));
	my_shader.setVec3("dirLight.specular", glm::vec3(1.0f, 1.0f, 1.0f));
	// point light 1
	my_shader.setVec3("pointLights[0].position", pointLightPositions[0]);
	my_shader.setVec3("lightPos[0]", pointLightPositions[0]);
	my_shader.setVec3("pointLights[0].ambient", 0.05f, 0.05f, 0.05f);
	my_shader.setVec3("pointLights[0].diffuse", 0.8f, 0.8f, 0.8f);
	my_shader.setVec3("pointLights[0].specular", 1.0f, 1.0f, 1.0f);
	my_shader.setFloat("pointLights[0].constant", 1.0f);
	my_shader.setFloat("pointLights[0].linear", 0.09);
	my_shader.setFloat("pointLights[0].quadratic", 0.032);
	// point light 2
	my_shader.setVec3("pointLights[1].position", pointLightPositions[1]);
	my_shader.setVec3("lightPos[1]", pointLightPositions[1]);
	my_shader.setVec3("pointLights[1].ambient", 0.05f, 0.05f, 0.05f);
	my_shader.setVec3("pointLights[1].diffuse", 0.8f, 0.8f, 0.8f);
	my_shader.setVec3("pointLights[1].specular", 1.0f, 1.0f, 1.0f);
	my_shader.setFloat("pointLights[1].constant", 1.0f);
	my_shader.setFloat("pointLights[1].linear", 0.09);
	my_shader.setFloat("pointLights[1].quadratic", 0.032);
	// point light 3
	my_shader.setVec3("pointLights[2].position", pointLightPositions[2]);
	my_shader.setVec3("lightPos[2]", pointLightPositions[2]);
	my_shader.setVec3("pointLights[2].ambient", 0.05f, 0.05f, 0.05f);
	my_shader.setVec3("pointLights[2].diffuse", 0.6f, 0.1f, 0.8f);
	my_shader.setVec3("pointLights[2].specular", 1.0f, 1.0f, 1.0f);
	my_shader.setFloat("pointLights[2].constant", 1.0f);
	my_shader.setFloat("pointLights[2].linear", 0.09);
	my_shader.setFloat("pointLights[2].quadratic", 0.032);
	// point light 4
	my_shader.setVec3("pointLights[3].position", pointLightPositions[3]);
	my_shader.setVec3("lightPos[3]", pointLightPositions[3]);
	my_shader.setVec3("pointLights[3].ambient", 0.05f, 0.05f, 0.05f);
	my_shader.setVec3("pointLights[3].diffuse", 0.1f, 1.1f, 0.8f);
	my_shader.setVec3("pointLights[3].specular", 1.0f, 1.0f, 1.0f);
	my_shader.setFloat("pointLights[3].constant", 1.0f);
	my_shader.setFloat("pointLights[3].linear", 0.09);
	my_shader.setFloat("pointLights[3].quadratic", 0.032);
	// normal map switch
	my_shader.setBool("useNormalMap", false);

	clock_t ltime = 0;
	clock_t ctime = 0;
	double duration = 0;
	float animT = 0.0f;

	// gui parameters
	bool show_demo_window = true;
	bool show_another_window = false;
	ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    while (!glfwWindowShouldClose(window))
    {
		if (playSwitch) {
			ctime = clock();
			duration = ((double)ctime - (double)ltime) / CLK_TCK;
			if (duration >= 1.0 / 30.0) {
				ltime = ctime;
				animT += 1.0 / 30.0;
				if (animT > animTime) {
					animT = 0.0f;
				}
				std::vector<float> NC = getNC(animT / animTime);
				Eigen::Vector2f res(0.0f, 0.0f);
				for (int i = 0; i <= curveN; i++) {
					res += NC[i] * curvePSet[i];
				}
				model_yaw = res[0];
				model_pitch = res[1];
			}
		}
		
        // input
        // -----
        processInput(window);

        // render
        // ------
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT| GL_DEPTH_BUFFER_BIT);
		//Update Camera Matrix
		glFlush();
		glEnable(GL_MULTISAMPLE);
		glEnable(GL_LIGHTING);
		glEnable(GL_COLOR_MATERIAL);
		glLightModeli(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
		float currentFrame = glfwGetTime();
		deltaTime = currentFrame - lastFrame;
		lastFrame = currentFrame;

		glm::mat4 projection = glm::perspective(0.785f, (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
		glm::mat4 view = glm::lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
		glm::mat4 model = glm::mat4(1.0f); // not used
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//  Render the lamp cubes                                                                                      //
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		lampShader.use();
		lampShader.setMat4("projection", projection);
		lampShader.setMat4("view", view);
		glBindVertexArray(cubeVAO);
		for (unsigned int i = 0; i < 4; i++)
		{
			model = glm::mat4(1.0f); // re-model
			model = glm::translate(model, pointLightPositions[i]);
			model = glm::scale(model, glm::vec3(0.2f)); // Make it a smaller cube
			lampShader.setMat4("model", model);
			glDrawArrays(GL_TRIANGLES, 0, 36);
		}
		
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//  Render an object using texture and normal map.                                                             //
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// bind Texture
		// mat
		//my_shader.use();

		// set sampler2D
		/*my_shader.setInt("mat.diffuse", 0);
		my_shader.setInt("mat.specular", 0);
		my_shader.setInt("normalMap", 2);*/

		// set texture unit ID
		/*glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture1);
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, textureNormal);

		my_shader.setMat4("projection", projection);
		my_shader.setMat4("view", view);
		model = glm::mat4(1.0f);
		my_shader.setMat4("model", model);
		my_shader.setVec3("viewPos", cameraPos);
		glDrawArrays(GL_TRIANGLES, 0, 36);*/


		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//  Render the object in .obj file. You need to set materials and wrap texture for objects.                    //
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		my_shader.use();
		my_shader.setInt("mat.diffuse", 1);
		my_shader.setInt("mat.specular", 1);
		//my_shader.setInt("normalMap", 2);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, texture1);
		glActiveTexture(GL_TEXTURE2);
		glBindTexture(GL_TEXTURE_2D, texture2);

		my_shader.setMat4("projection", projection);
		my_shader.setMat4("view", view);
		model = glm::mat4(1.0f);
		// translation are processed from bottom to up, at here,
		// first move to centre, then scale 0.1, pitch should be changed before yaw (important)..
		model = glm::rotate(model, glm::radians(model_yaw), glm::vec3(0.0f, 1.0f, 0.0f));
		model = glm::rotate(model, glm::radians(model_pitch), glm::vec3(1.0f, 0.0f, 0.0f));
		model = glm::scale(model, glm::vec3(0.1f));
		model = glm::translate(model, glm::vec3(1.7, -4.0, 0));
		my_shader.setMat4("model", model);
		my_shader.setVec3("viewPos", cameraPos);
		my_shader.setVec3("viewPos2VS", cameraPos);
		for (int i = 0; i < obj_VAO_l.size(); i++) {
			// body
			if (obj_VAO_l[i] == 3) {
				my_shader.setInt("mat.diffuse", 1);
				my_shader.setInt("mat.specular", 1);
			}
			// eye
			else {
				my_shader.setInt("mat.diffuse", 2);
				my_shader.setInt("mat.specular", 2);
			}
			glBindVertexArray(obj_VAO_l[i]);
			//printf("%d  %d\n", obj_VAO_l[i], obj_NUM_l[i]);
			glDrawArrays(GL_TRIANGLES, 0, obj_NUM_l[i]);
		}

		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////gui/////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Start the Dear ImGui frame
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		// 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
		if (show_demo_window)
			ImGui::ShowDemoWindow(&show_demo_window);
		// 2. Show a simple window that we create ourselves. We use a Begin/End pair to created a named window.
		{
			static float f = 0.0f;
			static int counter = 0;

			ImGui::Begin("Tool");                          // Create a window called "Hello, world!" and append into it.

			ImGui::Text("Use WASD to rotate model!");               // Display some text (you can use a format strings too)
			//ImGui::Checkbox("Demo Window", &show_demo_window);      // Edit bools storing our window open/close state
			ImGui::Checkbox("Ready", &show_another_window);

			ImGui::InputInt("first decide the degree of B-spine", &curveP);
			ImGui::InputFloat("s (then decide how long the animation lasts)", &animTime);
			ImGui::SliderFloat("s", &f, 0.0f, animTime);            // Edit 1 float using a slider from 0.0f to 1.0f
			ImGui::Text("Use the slide to specify any moment of the animation");
			//ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color
			if (ImGui::Button("Record")) {
				addD(Eigen::Vector2f(model_yaw, model_pitch), f / animTime);
				counter++;
			}
			ImGui::SameLine();
			if (ImGui::Button("Restart")) {
				playSwitch = false;
				restart();
				animT = 0.0f;
				counter = 0;
			}
			ImGui::SameLine();
			if (ImGui::Button("Sample")) {
				playSwitch = false;
				restart();
				animT = 0.0f;
				counter = 0;
				setSample();
				counter = 9;
			}
			//if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
			//	counter++;
			ImGui::SameLine();
			//ImGui::Text("counter = %d", counter);
			ImGui::Text("%d key frames recorded.", counter);
			ImGui::End();
		}
		// 3. Show another simple window.
		if (show_another_window)
		{
			ImGui::Begin("Player", &show_another_window);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
			ImGui::Text("click button to play animation!");
			if (ImGui::Button("Play!")) {
				getP();
				playSwitch = true;
			}
			if (ImGui::Button("Stop!")) {
				playSwitch = false;
			}	
			ImGui::End();
		}
		// Rendering
		ImGui::Render();
		int display_w, display_h;
		glfwGetFramebufferSize(window, &display_w, &display_h);
		glViewport(0, 0, display_w, display_h);
		// remove this otherwise it will clear former rendering
		//glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
		//glClear(GL_COLOR_BUFFER_BIT);
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////gui/////////////////////////////////////////////////////////////////////////
		//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		
		/////////////////////////////////////////////////////////////////////
		
		/////////////////////////////end/////////////////////////////////////

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
	glDeleteVertexArrays(1, &cubeVAO);
	glDeleteBuffers(1, &cubeVBO);
	for (int i = 0; i < obj_VAO_l.size(); i++) {
		glDeleteVertexArrays(1, &obj_VAO_l[i]);
		glDeleteBuffers(1, &obj_VBO_l[i]);
	}

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

