import torch
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from datetime import datetime
from tablewarenet.tableware import name_to_class

class AppWindow:

	def __init__(self):

		# initialize
		self.category_idx = 0
		self.category = 'WineGlass'
		self.shape_parameters = torch.zeros(10)

		# parameters 
		image_size = [1024, 768]

		# object material
		self.mat = rendering.MaterialRecord()
		self.mat.shader = 'defaultLit'
		self.mat.base_color = [1.0, 1.0, 1.0, 0.9]
		mat_prev = rendering.MaterialRecord()
		mat_prev.shader = 'defaultLitTransparency'
		mat_prev.base_color = [1.0, 1.0, 1.0, 0.7]
		mat_coord = rendering.MaterialRecord()
		mat_coord.shader = 'defaultLitTransparency'
		mat_coord.base_color = [1.0, 1.0, 1.0, 0.87]

		######################################################
		################# STARTS FROM HERE ###################
		######################################################

		# set window
		self.window = gui.Application.instance.create_window(
			str(datetime.now().strftime('%H%M%S')), 
			width=image_size[0], height=image_size[1])
		w = self.window
		self._scene = gui.SceneWidget()
		self._scene.scene = rendering.Open3DScene(w.renderer)

		# camera viewpoint
		self._scene.scene.camera.look_at(
			[0, 0, 0], # camera lookat
			[0.7, 0, 0.9], # camera position
			[0, 0, 1] # fixed
		)

		# other settings
		self._scene.scene.set_lighting(
			self._scene.scene.LightingProfile.DARK_SHADOWS, 
			(-0.3, 0.3, -0.9))
		self._scene.scene.set_background(
			[1.0, 1.0, 1.0, 1.0], 
			image=None)

		############################################################
		######################### MENU BAR #########################
		############################################################
		
		# menu bar initialize
		em = w.theme.font_size
		separation_height = int(round(0.5 * em))
		self._settings_panel = gui.Vert(
			0, gui.Margins(0.25 * em, 0.25 * em, 0.25 * em, 0.25 * em))

		# initialize collapsable vert
		dataset_config = gui.CollapsableVert(
			"Dataset config", 0.25 * em, gui.Margins(em, 0, 0, 0))

		# Object category
		self._wineglass_button = gui.Button("Wineglass")
		self._wineglass_button.horizontal_padding_em = 0.5
		self._wineglass_button.vertical_padding_em = 0
		self._wineglass_button.set_on_clicked(self._set_object_wineglass)
		self._bowl_button = gui.Button("Bowl")
		self._bowl_button.horizontal_padding_em = 0.5
		self._bowl_button.vertical_padding_em = 0
		self._bowl_button.set_on_clicked(self._set_object_bowl)
		self._bottle_button = gui.Button("Bottle")
		self._bottle_button.horizontal_padding_em = 0.5
		self._bottle_button.vertical_padding_em = 0
		self._bottle_button.set_on_clicked(self._set_object_bottle)
		h = gui.Horiz(0.25 * em)
		h.add_stretch()
		h.add_child(self._wineglass_button)
		h.add_child(self._bowl_button)
		h.add_child(self._bottle_button)
		h.add_stretch()

		# add
		dataset_config.add_child(gui.Label("Object category"))
		dataset_config.add_child(h)

		# Object category 2
		self._beerbottle_button = gui.Button("BeerBottle")
		self._beerbottle_button.horizontal_padding_em = 0.5
		self._beerbottle_button.vertical_padding_em = 0
		self._beerbottle_button.set_on_clicked(self._set_object_beerbottle)
		self._handlesscup_button = gui.Button("HandlessCup")
		self._handlesscup_button.horizontal_padding_em = 0.5
		self._handlesscup_button.vertical_padding_em = 0
		self._handlesscup_button.set_on_clicked(self._set_object_handlesscup)
		self._mug_button = gui.Button("Mug")
		self._mug_button.horizontal_padding_em = 0.5
		self._mug_button.vertical_padding_em = 0
		self._mug_button.set_on_clicked(self._set_object_mug)
		h = gui.Horiz(0.25 * em)
		h.add_stretch()
		h.add_child(self._beerbottle_button)
		h.add_child(self._handlesscup_button)
		h.add_child(self._mug_button)
		h.add_stretch()

		# add
		dataset_config.add_child(h)

		# Object category 3
		self._dish_button = gui.Button("Dish")
		self._dish_button.horizontal_padding_em = 0.5
		self._dish_button.vertical_padding_em = 0
		self._dish_button.set_on_clicked(self._set_object_dish)
		h = gui.Horiz(0.25 * em)
		h.add_stretch()
		h.add_child(self._dish_button)
		h.add_stretch()

		# add
		dataset_config.add_child(h)
		dataset_config.add_fixed(separation_height)

		##########################################
		################ INITIALIZE ##############
		##########################################

		# name list
		self.category_name_list = [
			'WineGlass', 'Bowl', 'Bottle', 
			'BeerBottle', 'HandlessCup', 'Mug', 
			'Dish']
		# self.category_name_list = ['WineGlass']

		# config initialize
		self.config_list = [
			gui.CollapsableVert(
				name, 0.25 * em, gui.Margins(em, 0, 0, 0)
			) for name in self.category_name_list
		]

		# callback function initialize
		self._set_shape_parameter_list = [
			self._set_shape_parameter_0,
			self._set_shape_parameter_1,
			self._set_shape_parameter_2,
			self._set_shape_parameter_3,
			self._set_shape_parameter_4,
			self._set_shape_parameter_5,
			self._set_shape_parameter_6,
			self._set_shape_parameter_7,
			self._set_shape_parameter_8,
			self._set_shape_parameter_9,
		]

		self._shape_parameter_slider_list = []

		##########################################
		################ CATEGORIES ##############
		##########################################

		for i, (config, name) in enumerate(zip(self.config_list, self.category_name_list)):

			# collapse config
			config.set_is_open(False)
			
			# parameter initialize
			param_dict = name_to_class[name](
				torch.eye(4), params='random', device='cpu'
			).range

			# initialize slider list
			self._objectwise_slider_list = []

			# parameter sliders
			for i, (param_name, param_range) in enumerate(param_dict.items()):
				
				# slider
				self._silder = gui.Slider(gui.Slider.DOUBLE)
				self._silder.set_limits(param_range[0], param_range[1])
				self._silder.double_value = (
					param_range[0] + param_range[1]) / 2
				self._silder.set_on_value_changed(
					self._set_shape_parameter_list[i])

				# add
				config.add_child(gui.Label(param_name))
				config.add_child(self._silder)         

				# append
				self._objectwise_slider_list.append(self._silder)

			# append
			self._shape_parameter_slider_list.append(self._objectwise_slider_list)

		# add 
		self._settings_panel.add_child(dataset_config)
		for config in self.config_list:
			self._settings_panel.add_child(config)

		##########################################
		################ ADD CHILD ###############
		##########################################

		# add scene
		w.set_on_layout(self._on_layout)
		w.add_child(self._scene)
		w.add_child(self._settings_panel)

	############################################################
	######################### FUNCTIONS ########################
	############################################################

	def _on_layout(self, layout_context):
		r = self.window.content_rect
		self._scene.frame = r
		width = 17 * layout_context.theme.font_size
		height = min(
			r.height,
			self._settings_panel.calc_preferred_size(
				layout_context, gui.Widget.Constraints()).height)
		self._settings_panel.frame = gui.Rect(r.get_right() - width, r.y, width,
											  height)

	def initialize_object(self):

		# clear geometry
		self._scene.scene.clear_geometry()
		for config in self.config_list:
			config.set_is_open(False)

		# set shape category
		self.category = self.category_name_list[self.category_idx]
		self.config_list[self.category_idx].set_is_open(True)
		self.window.set_needs_layout()

		# set initial parameters
		slider_list = self._shape_parameter_slider_list[self.category_idx]
		for i, slider in enumerate(slider_list):
			self.shape_parameters[i] = slider.double_value

		# get object
		obj = name_to_class[self.category](
			torch.eye(4), params=self.shape_parameters, device='cpu'
		)

		# get mesh
		mesh = obj.get_mesh()

		# add 
		self._scene.scene.add_geometry(f'object', mesh, self.mat)

	def update_object(self):
			
		# clear geometry
		self._scene.scene.clear_geometry()

		# get object
		obj = name_to_class[self.category](
			torch.eye(4), params=self.shape_parameters, device='cpu'
		)

		# get mesh
		mesh = obj.get_mesh()

		# add 
		self._scene.scene.add_geometry(f'object', mesh, self.mat)

	def _set_object_wineglass(self):
		self.category_idx = 0
		self.initialize_object()

	def _set_object_bowl(self):
		self.category_idx = 1
		self.initialize_object()

	def _set_object_bottle(self):
		self.category_idx = 2
		self.initialize_object()

	def _set_object_beerbottle(self):
		self.category_idx = 3
		self.initialize_object()

	def _set_object_handlesscup(self):
		self.category_idx = 4
		self.initialize_object()

	def _set_object_mug(self):
		self.category_idx = 5
		self.initialize_object()

	def _set_object_dish(self):
		self.category_idx = 6
		self.initialize_object()

	def _set_shape_parameter_0(self, value):
		self.shape_parameters[0] = value
		self.update_object()

	def _set_shape_parameter_1(self, value):
		self.shape_parameters[1] = value
		self.update_object()

	def _set_shape_parameter_2(self, value):
		self.shape_parameters[2] = value
		self.update_object()

	def _set_shape_parameter_3(self, value):
		self.shape_parameters[3] = value
		self.update_object()

	def _set_shape_parameter_4(self, value):
		self.shape_parameters[4] = value
		self.update_object()

	def _set_shape_parameter_5(self, value):
		self.shape_parameters[5] = value
		self.update_object()

	def _set_shape_parameter_6(self, value):
		self.shape_parameters[6] = value
		self.update_object()

	def _set_shape_parameter_7(self, value):
		self.shape_parameters[7] = value
		self.update_object()

	def _set_shape_parameter_8(self, value):
		self.shape_parameters[8] = value
		self.update_object()

	def _set_shape_parameter_9(self, value):
		self.shape_parameters[9] = value
		self.update_object()

if __name__ == "__main__":

	# # web visualizer
	# o3d.visualization.webrtc_server.enable_webrtc()
	
	# run
	gui.Application.instance.initialize()
	w = AppWindow()
	gui.Application.instance.run()