#!/usr/bin/python
# -*- coding:utf8 -*-
import hydra
import os
import json

import torch
# from control.control_service import control_service_start

# from control.utils import my_JSON_serializable, dict_to_Tensor

import requests
import time
from utils.common import SimpleLogger
from control.thickener_pressure_simulation import ThickenerPressureSimulation
from control.industrial_benchmark_simulation import IndustrialBenchmarkSimulation
from control.pid_thickener_pressure_simulation import PIDThickenerPressureSimulation
from omegaconf import DictConfig

# parser = argparse.ArgumentParser('Pressure control Test')
# parser.add_argument('-R',  type=int, default=400, help="Rounds for Test")
# parser.add_argument('--simulation', type=str, default='control/trained_model/cstr_vrnn_5_cpu.pkl', help='ckpt path of simulation trained_model.')
# parser.add_argument('--planning',  type=str, default='control/trained_model/cstr_vrnn_5_cpu.pkl', help="ckpt path of planning trained_model.")
# parser.add_argument('--ip',  type=str, default='localhost', help="ckpt path of planning trained_model.")
# parser.add_argument('-v', '--vis', action='store_true', default=False)
# parser.add_argument('--service', action='store_true', default=False)
# parser.add_argument('-cuda',  type=int, default=0, help="GPU ID")
# parser.add_argument('--length',  type=int, default=50, help="The length of optimized sequence for planning")
# parser.add_argument('--num_samples',  type=int, default=32, help="The number of samples in CEM planning")
# parser.add_argument('--num_iters',  type=int, default=32, help="The number of iters in CEM planning")
# parser.add_argument('-r', '--random_seed',  type=int, default=1, help="Random seed in experiment")
# parser.add_argument('-dataset', type=str, default='./data/southeast', help="The simulated dateset")
# parser.add_argument('-input_dim', type=int, default=1, help="output_dim of trained_model")
# parser.add_argument('-output_dim', type=int, default=2, help="input_dim of trained_model")
# parser.add_argument('--set_value', type=list, default=[0.8,0.1,0.5], help='The set_value of control')  # [number of output_dim; number of input_dim]
# parser.add_argument('--port',  type=int, default=6010, help="The number of iters in CEM planning")
# parser.add_argument('--debug', action='store_true', default=False)
# config = parser.parse_args()


def get_ob_list(cur_ob_dict):
    cur_ob = [
        cur_ob_dict['observation'],
        cur_ob_dict['action'],
    ]
    return [cur_ob]  # ??????????????????????????????????????????????????????

# # ?????????
# def scale(action):
#
#     action = (action - trained_model.scale[0])  /  trained_model.scale[1]
#     return action
#
# # ????????????
# def unscale(obs):
#     obs = obs * trained_model.scale[1] - trained_model.scale[0]
#     return obs


@hydra.main(config_path='config', config_name='config_test.yaml')
def main(args:DictConfig):
    device = torch.device("cuda:{}".format(str(args.cuda)) if torch.cuda.is_available() else "cpu")

    logging = SimpleLogger('./log.out')
    figs_path = os.path.join(os.getcwd(), 'control/figs/',
                                  time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime(time.time())))
    if not os.path.exists(figs_path):
        os.makedirs(figs_path)

    if args.modeltype == 'darts':
        from darts.models import BlockRNNModel

        model = BlockRNNModel.load_model(os.path.join(hydra.utils.get_original_cwd(), 'trained_model', args.model))
    else:
        model = torch.load(os.path.join(hydra.utils.get_original_cwd(), 'trained_model', args.model), map_location={'cuda:0': 'cuda:2'})
        model = model.to(device)

    logging('save dir = {}'.format(os.getcwd()))
    # ?????????????????????
    if args.use_benchmark and args.benchmark == "ib":
        simulated_thickener = IndustrialBenchmarkSimulation(args.set_value, args.input_dim, args.output_dim, args.used_columns,
                                                          figs_path=figs_path,
                                                          model=model,
                                                          dataset_path=args.dataset,
                                                          random_seed=args.random_seed, device=device)
    elif args.use_benchmark and args.benchmark == "pid":
        simulated_thickener = PIDThickenerPressureSimulation(args.set_value, args.input_dim, args.output_dim, args.used_columns,
                                                          figs_path=figs_path,
                                                          model=model,
                                                          random_seed=args.random_seed, device=device)
    else:
        simulated_thickener = ThickenerPressureSimulation(args.set_value, args.input_dim, args.output_dim, args.used_columns,
                                                          figs_path=figs_path,
                                                          model=model,
                                                          random_seed=args.random_seed, device=device)

    logging('simulation thickener have been built')
    # region ?????????????????????????????????????????????memory_state
    resp = requests.post(
        "http://{}:{}/update".format(args.ip, str(args.port)),
        data={
            'monitoring_data': json.dumps(get_ob_list(simulated_thickener.get_current_state())),
            'memory_state': None
        })
    memory_state_json = resp.json()['memory_state']
    # endregion
    for _ in range(args.R):

        # region ????????????cem-planning
        resp = requests.post("http://{}:{}/planning".format(args.ip, str(args.port)), data={
            'memory_state': memory_state_json,
            'time': _
        })
        print('planning - {}'.format(resp.json()))
        planning_action = resp.json()['planning_action']

        # ???????????????????????????????????????
        simulated_thickener.step(planning_action)
        # region ???????????????????????????????????????memory-state
        resp = requests.post("http://{}:{}/update".format(args.ip, str(args.port)), data={
            'monitoring_data': json.dumps(get_ob_list(simulated_thickener.get_current_state())),   # ?????????????????????
            'memory_state': memory_state_json,
        })
        memory_state_json = resp.json()['memory_state']
        print('update - resp={}'.format(resp.json()))
        # endregion



if __name__ == '__main__':
    # from common import SimpleLogger
    # logging = SimpleLogger('./log.out')
    # main(config,logging)
    main()
