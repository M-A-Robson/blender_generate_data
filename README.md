# blender_generate_data
blender files for creating set of object views on a table with masks, keypoints and various python helpers

clamp_on_table.blend is an example blender world with a table object and one clamp object placed on it.
 - The main scene holds the clamp and table objects
 - scene001 holds spheres which represent keypoints of the object. These can be rendered out on different passes (this seems to be limited to cpu rendering) to capture images with each keypoint as a circle.

keypoint_cameras.py can be ran in the blender scripting tool to generate camera positions for cameras placed on the verticies of the sphere object, looking into the scene. The script should be run after any edits of the sphere position, radius, or number of vertices, to regenerate camera keyframes.
Use render animation to render all frames. (you need to set the save path in the file out node of the compositor first)

The code file contains helper scripts for post processing:
- convert_exr_to_np.py - extracts numpy int array data from the OpenEXR depth files
- extract_circle_centers.py - converts the keypoint images into numpy arrays of shape (n,x,y) of the keypoint positions in each image pose (the original kp_i_j images can then be deleted saving a large amount of storage)
- inject_random_image_noise.py - provides a noisify function which can be used to add a range of synthetic noise to the created images, see the code for more details.


