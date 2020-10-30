#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
//Do not use image bigger than 30kB. Spuare image is better
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Here you need to fill the loadTexture function, so that you can use it to load texture in main.             //
// width and height must be pow of 2                                                                           //
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
unsigned int loadTexture(char const * path)
{
	unsigned int textureID;
	// number of textures
	glGenTextures(1, &textureID);
	int width, height, nrComponents;
	unsigned char *data = stbi_load(path, &width, &height, &nrComponents, 0);

	if (data != NULL)
	{
		// bind to 2D target
		glBindTexture(GL_TEXTURE_2D, textureID);
		// basic level 0; RGB format store; RGB format source; data type; real picture data 
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
		// automatically generate mipmap(or manually set all level increasing the second parameter above)
		glGenerateMipmap(GL_TEXTURE_2D);
		// will repeat if uv exceed range0-1
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);// S axis, repeat
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);// T axis, repeat
		
		// linear interpolation between two near level and than sample with linear interpolation
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR); // when texture minified
		// linear filter out a color from surrounding pixel
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);// when texture magnified

	}
	else {
		printf("load texture failed\n");
	}
	// free data after texture creation
	stbi_image_free(data);

	return textureID;
}