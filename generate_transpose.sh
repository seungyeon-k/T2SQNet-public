#!/bin/bash

BLENDER_BIN=~/blender-4.0.2-linux-x64/./blender
BLENDER_PROJ_PATH=assets/materials/material_lib_graspnet-v2.blend

$BLENDER_BIN $BLENDER_PROJ_PATH --background \
--python transpose_dataset_generation.py \
--- \
---num_objects 4 \
---sim_type table \
---object_types TRansPose \
---folder_name TRansPose_4_0524 \
---num_cameras 36 \
---data_num 50 \
---use_blender
# ---enable_gui \
# ---debug