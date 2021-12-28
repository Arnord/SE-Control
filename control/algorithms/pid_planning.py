#!/usr/bin/python
# -*- coding:utf8 -*-

import numpy as np
import os
import torch

#####
from model.func import normal_differential_sample
from model.common import DiagMultivariateNormal as MultivariateNormal
#####


class PIDPlanning:

    def __init__(self, input_dim, output_dim, length, dt, max, min, KP, KI, KD, device='cpu', time=0):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.length = length
        self.device = device
        self.dt = dt
        self.max = max
        self.min = min
        self.KP = KP
        self.KI = KI
        self.KD = KD

        self.figs_path = os.path.join(os.getcwd(), 'control/pid_figs')
        if not os.path.exists(self.figs_path):
            os.makedirs(self.figs_path)
        self.test = False
        self.pid_time = int(time)

        self.ATC = 0.6        # attenuation coefficient of [output - last_output]

        print("pid init over")

    def solve(self, setPoint, memory_state, op_var=None):
        """

        Args:
            setPoint: 设定值
            pv: process value 过程值

        Returns:
        :param memory_state:
        :param setPoint:
        :param op_var:

        """
        if op_var is None:
            op_var = dict(
                sum_err=0,
                last_err=0,
                last_output=50)

        sum_err = op_var['sum_err']
        last_err = op_var['last_err']
        last_output = op_var['last_output']

        if self.test:
            return 0
        else:
            exp_val_c = float(setPoint)
            now_val_c = memory_state['monitoring_data'][:self.output_dim][0]

            error = exp_val_c - now_val_c  # 误差
            p_out = self.KP * error  # 比例项
            sum_err += error * self.dt
            i_out = self.KI * sum_err  # 积分项
            derivative = (error - last_err) / self.dt  # 微分项
            d_out = self.KD * derivative
            pv_out = p_out + i_out + d_out

            output = last_output + self.ATC * (pv_out - last_output)

            if output > self.max:
                output = self.max
            elif output < self.min:
                output = self.min

            if isinstance(output, list):
                pass
            else:
                output = [output]
            last_err = error

            op_var['sum_err'] = sum_err
            op_var['last_err'] = last_err
            op_var['last_output'] = output
            new_op_var = op_var

        return new_op_var, output
