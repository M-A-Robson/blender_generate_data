import bpy

vp_obj = bpy.data.objects['Sphere']
cam_obj = bpy.data.objects['Camera']
cam_obj_1 = bpy.data.objects['Camera.001']
vp_vs = vp_obj.data.vertices

bpy.context.scene.frame_start=1
bpy.context.scene.frame_end=len(vp_vs)

for (f,v) in enumerate(vp_vs,1):
    cam_obj.location = vp_obj.matrix_world * v.co
    cam_obj.keyframe_insert(data_path="location", frame=f)
    cam_obj_1.location = vp_obj.matrix_world * v.co
    cam_obj_1.keyframe_insert(data_path="location", frame=f)