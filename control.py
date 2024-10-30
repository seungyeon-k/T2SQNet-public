import argparse
from omegaconf import OmegaConf

from control.controller import Controller
from utils.yaml_utils import parse_unknown_args, parse_nested_args

if __name__ == "__main__":
	
	# argparse
	parser = argparse.ArgumentParser()
	parser.add_argument('--config', type=str)
	parser.add_argument('--device', default=0)
	parser.add_argument('--debug', action='store_true')

	# process cfg
	args, unknown = parser.parse_known_args()
	d_cmd_cfg = parse_unknown_args(unknown)
	d_cmd_cfg = parse_nested_args(d_cmd_cfg)
	cfg = OmegaConf.load(args.config)
	cfg = OmegaConf.merge(cfg, d_cmd_cfg)

	# set device
	if args.device == 'cpu':
		cfg.device = 'cpu'
	elif args.device == 'any':
		cfg.device = 'cuda'
	else:
		cfg.device = f'cuda:{args.device}'

	# set debug
	cfg.debug = args.debug

	# control
	controller = Controller(cfg)
	if cfg.task_type == 'clear_clutter':
		controller.clear_clutter()
	elif cfg.task_type == 'target_retrieval':
		controller.target_retrieval()
	else:
		raise ValueError(f'task_type {cfg.task_type} not in ["clear_clutter", "target_retrieval"]')