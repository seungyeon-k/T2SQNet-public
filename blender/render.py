import os
import random
import torch
import bpy
import math
import numpy as np
from blender.modify_material import (
    set_modify_table_material, set_modify_floor_material)
from blender.render_utils import *
from functions.utils_torch import rotation_matrices_to_quaternions_torch
from copy import deepcopy

def blender_init_scene(
        asset_root, 
        data_path, 
        obj_texture_image_root_path, 
        urdfs_and_poses_dict, 
        round_idx, 
        gpuid, 
        camera_image_size,
        camera_view_poses,
        sim_type,
        use_background=True,
        material_debug=False,
        background_idx=None,
        table_idx=None,
        glass_idx=None):

    # set background parameter
    background_size = 10.
    background_position = (0., 0., 0.)
    background_scale = (1., 1., 1.)

    # initialize
    DEVICE_LIST = [int(gpuid)]

    # load blender env
    obj_texture_image_idxfile = "test_paths.txt"
    env_map_path = os.path.join(
        asset_root, "envmap_lib_test")
    real_table_image_root_path = os.path.join(
        asset_root, "realtable_test")
    real_floor_image_root_path = os.path.join(
        asset_root, "realfloor_test")
    emitter_pattern_path = os.path.join(
        asset_root, "pattern", "test_pattern.png")
    default_background_texture_path = os.path.join(
        asset_root, "texture", "texture_0.jpg")

    # load table and shelf (should be fixed to be loaded from the env)
    high_table_path = os.path.join(
        'assets', 'table', 'mesh', 'high_table.obj'
    )
    high_table_position = [0.61, -0.0375, 0.243-0.05/2]
    shelf_path = os.path.join(
        'assets', 'shelf', 'mesh', 'shelf.obj'
    )
    shelf_position = [0.61, 0., 0.003]

    # log directory
    output_root_path = os.path.join(
        data_path, 'rendered_results', str(round_idx))
    if not os.path.exists(output_root_path):
        os.makedirs(output_root_path)

    # load object information
    obj_uid_list = [str(uid) for uid in urdfs_and_poses_dict]
    obj_scale_list = [value[0] for value in urdfs_and_poses_dict.values()]
    obj_quat_list = [value[1][[3, 0, 1, 2]] for value in urdfs_and_poses_dict.values()]
    obj_trans_list = [value[2] for value in urdfs_and_poses_dict.values()]
    urdf_path_list = [os.path.join(value[3]) for value in urdfs_and_poses_dict.values()]

    # generate CAD model list
    select_model_list = generate_CAD_model_list(
        urdf_path_list, obj_uid_list)

    # generate camara pose list
    translation_list = []
    quaternion_list = []
    for i in range(len(camera_view_poses)):
        camera_pose = deepcopy(camera_view_poses[i]) 
        camera_ori = camera_pose[:3, :3]
        camera_ori[:, 1:] *= -1 # open3d to blender
        
        # append
        translation_list.append(camera_pose[:3, 3])
        quaternion_list.append(
            rotation_matrices_to_quaternions_torch(
                torch.tensor(camera_ori).unsqueeze(0)
            )[0].numpy().astype(float)
        )

    # blender renderer
    camera_width = camera_image_size[1]
    camera_height = camera_image_size[0]
    renderer = BlenderRenderer(
        viewport_size_x=camera_width, 
        viewport_size_y=camera_height, 
        DEVICE_LIST=DEVICE_LIST
    )
    renderer.loadImages(
        emitter_pattern_path, 
        env_map_path, 
        real_table_image_root_path, 
        real_floor_image_root_path, 
        obj_texture_image_root_path, 
        obj_texture_image_idxfile, 
        False)
    renderer.addEnvMap()
    renderer.addMaterialLib()
    # renderer.addMaskMaterial(20)
    # renderer.addNOCSMaterial()
    # renderer.addNormalMaterial()
    renderer.clearModel()

    # set environment map
    env_map_id_list = []
    rotation_elur_z_list = []
    if material_debug:
        flag_env_map = 0
        flag_env_map_rot = 0.0
    else:
        flag_env_map = random.randint(0, len(renderer.env_map) - 1)
        flag_env_map_rot = random.uniform(-math.pi, math.pi)
    env_map_id_list.append(flag_env_map)
    rotation_elur_z_list.append(flag_env_map_rot)
    renderer.setEnvMap(env_map_id_list[0], rotation_elur_z_list[0])

    # background
    if use_background:
        renderer.addBackground(
            background_size, 
            background_position, 
            background_scale, 
            default_background_texture_path)
        # material_selected = renderer.my_material['default_background']
        if material_debug:
            material_selected = renderer.my_material['background'][0]
            flag_realfloor = 0
        else:
            if background_idx is None:
                material_selected = random.sample(
                    renderer.my_material['background'], 1)[0]
            else:
                material_selected = renderer.my_material['background'][background_idx]
            flag_realfloor = random.randint(
                0, len(renderer.realfloor_img_list) - 1)
        selected_realfloor_img = renderer.realfloor_img_list[flag_realfloor]
        for obj in bpy.data.objects:
            if obj.type == "MESH" and obj.name.split('_')[0] == 'background':
                if obj.name == 'background_0':
                    set_modify_floor_material(obj, material_selected, selected_realfloor_img)
                else:
                    background_0_obj = bpy.data.objects['background_0']
                    obj.active_material = background_0_obj.material_slots[0].material

    # set table or shelf
    if sim_type == 'table':
        # table model
        renderer.loadModel(high_table_path)
        high_table_obj = bpy.data.objects['high_table']
        high_table_orientation = quaternionFromRotMat(np.eye(3))
        setModelPosition(
            high_table_obj, high_table_position, high_table_orientation)

        # set material
        # material_selected = random.sample(
        #     renderer.my_material['glass'], 1)[0]
        if material_debug:
            material_selected = renderer.my_material['wood'][0]
        else:
            if table_idx is None:
                material_selected = random.sample(
                    renderer.my_material['wood'], 1)[0]
            else:
                material_selected = renderer.my_material['wood'][table_idx]
        high_table_obj.active_material = material_selected

        # # additional material settings
        # material_selected = random.sample(
        #     renderer.my_material['wood'], 1)[0]
        # selected_realtable_img = renderer.realtable_img_list[
        #     random.randint(0, len(renderer.realtable_img_list) - 1)]
        # set_modify_table_material(
        #     high_table_obj, material_selected, selected_realtable_img)

    elif sim_type == 'shelf':
        
        # shelf model
        renderer.loadModel(shelf_path)
        shelf_obj = bpy.data.objects['shelf']
        shelf_orientation = [0.0, 0.0, 0.0, 1.0]
        setModelPosition(
            shelf_obj, shelf_position, shelf_orientation)

        # set material
        if material_debug:
            material_selected = renderer.my_material['wood'][0]
        else:
            material_selected = random.sample(
                renderer.my_material['wood'], 1)[0]
        shelf_obj.active_material = material_selected

    else:
        raise ValueError('Check sim_type variable!!')
    
    # object models
    for instance_id, model in enumerate(select_model_list):
        
        # load info
        instance_path = model[0]
        class_name = model[1]
        class_scale = obj_scale_list[instance_id]
        instance_uid = model[2]
        instance_folder = model[0].split('/')[-1][:-4] 
        instance_name = str(instance_id) + "_" + class_name + "_" + instance_folder + "_" + instance_uid

        # download CAD model and rename
        renderer.loadModel(instance_path)
        obj = bpy.data.objects[instance_folder]
        obj.name = instance_name
        obj.data.name = instance_name
        obj.scale = (class_scale, class_scale, class_scale)
        setModelPosition(obj, obj_trans_list[instance_id], obj_quat_list[instance_id])

        # set object as rigid body
        setRigidBody(obj)

        # set material
        if material_debug:
            try:
                material_selected = renderer.my_material['glass'][round_idx]
            except:
                raise ValueError(f"round_idx is bigger than {len(renderer.my_material['glass'])-1}")
        else:
            if glass_idx is None:
                material_selected = random.sample(
                    renderer.my_material['glass'], 1)[0]
            else:
                material_selected = renderer.my_material['glass'][glass_idx]
        obj.active_material = material_selected

    return renderer, quaternion_list, translation_list, output_root_path

def blender_render(
        renderer, 
        quaternion_list, 
        translation_list, 
        path_scene, 
        render_frame_list, 
        output_modality_dict, 
        camera_focal, 
        is_init=False,
        baseline_distance=0.055):
    
    # set the key frame
    scene = bpy.data.scenes['Scene']

    # camera parameter
    camera_width = renderer.viewport_size_x
    camera_fov = 2 * math.atan(camera_width / (2 * camera_focal))

    # dictionary initialize
    save_name_list_dict = dict()
    for mod, b in output_modality_dict.items():
        if b:
            save_name_list_dict[mod] = list()
        
    # render IR image and RGB image
    if output_modality_dict['IR'] or output_modality_dict['RGB']:
        if is_init:
            renderer.src_energy_for_rgb_render = bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[1].default_value

        for i in range(len(quaternion_list)):  
            renderer.setCamera(
                quaternion_list[i], 
                translation_list[i], 
                camera_fov, 
                baseline_distance)
            renderer.setLighting()

            # render RGB image
            if output_modality_dict['RGB']:
                # rgb_dir_path = os.path.join(path_scene, 'rgb') # TODO: CHANGE IF WE USE MORE IMAGE TYPES
                rgb_dir_path = path_scene
                if os.path.exists(rgb_dir_path) == False:
                    os.makedirs(rgb_dir_path)

                renderer.render_mode = "RGB"
                camera = bpy.data.objects['camera_l']
                scene.camera = camera
                save_path = rgb_dir_path
                save_name = str(i).zfill(4)
                renderer.render(save_name, save_path)
                save_name_list_dict['RGB'].append(f'{save_name}.png')

            # render IR image
            if output_modality_dict['IR']:
                ir_l_dir_path = os.path.join(path_scene, 'ir_l')
                if os.path.exists(ir_l_dir_path)==False:
                    os.makedirs(ir_l_dir_path)
                ir_r_dir_path = os.path.join(path_scene, 'ir_r')
                if os.path.exists(ir_r_dir_path)==False:
                    os.makedirs(ir_r_dir_path)

                renderer.render_mode = "IR"
                camera = bpy.data.objects['camera_l']
                scene.camera = camera
                save_path = ir_l_dir_path
                save_name = str(i).zfill(4)
                renderer.render(save_name, save_path)

                camera = bpy.data.objects['camera_r']
                scene.camera = camera
                save_path = ir_r_dir_path
                save_name = str(i).zfill(4)
                renderer.render(save_name, save_path)

    # render normal map
    if output_modality_dict['Normal']:
        
        # set normal as material
        for obj in bpy.data.objects:
            if obj.type == 'MESH':
                obj.data.materials.clear()
                obj.active_material = renderer.my_material["normal"]

        # render normal map
        for i in range(len(quaternion_list)): 
            renderer.setCamera(
                quaternion_list[i], 
                translation_list[i], 
                camera_fov, 
                baseline_distance)

            normal_dir_path = os.path.join(path_scene, 'normal')
            if os.path.exists(normal_dir_path)==False:
                os.makedirs(normal_dir_path)
            depth_dir_path = os.path.join(path_scene, 'depth')
            if os.path.exists(depth_dir_path)==False:
                os.makedirs(depth_dir_path)

            renderer.render_mode = "Normal"
            camera = bpy.data.objects['camera_l']
            scene.camera = camera
            save_path = normal_dir_path
            save_name = str(i).zfill(4)
            renderer.render(save_name, save_path)

    context = bpy.context
    for ob in context.selected_objects:
        ob.animation_data_clear()

    return save_name_list_dict