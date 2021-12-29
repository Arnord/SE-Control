#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import os
import random
import pandas as pd
import datetime
from matplotlib import pyplot as plt
from darts import TimeSeries
from utils.dataframe_scalar import DataFrameScalar
from utils.arrayscalar import arrayScalar


class ThickenerSimulation:
    def __init__(self, random_seed, model, scalar, in_len=60, out_len=30):
        self.input_length = in_len
        self.output_length = out_len
        self.random_seed = random.randint(0, 1000000) if random_seed is None else random_seed
        self.model = model
        self.scalar = scalar

        self.start_time = datetime.datetime.now()

    def simulate(self, external_input, covariable):
        # 归一化
        external_input = self.scalar.scale_scalar(external_input, [1, 4])
        covariable = self.scalar.scale_scalar(covariable, [0, 2, 3])
        external_input = TimeSeries.from_times_and_values(
            pd.date_range(start=self.start_time, freq='T', periods=len(external_input)), external_input)
        covariable = TimeSeries.from_times_and_values(
            pd.date_range(start=self.start_time, freq='T', periods=len(covariable)), covariable)

        pred_out_f = self.model.predict(self.output_length, external_input,
                                        covariable)  # pred_out 为长度为self.output_chunk_length的TimeSeries
        # pred_out 逆归一化
        pred_out_f = self.scalar.unscale_scalar(pred_out_f.values(), [1, 4])
        return pred_out_f

    def plot(self, actual_in, actual_out, actual_pred):
        assert len(actual_out) == len(actual_pred)
        plt.plot(range(0, len(actual_in)), actual_in,
                 label='actual_in')  # 仿真浓密机数据  -----暂时只选择pressure
        plt.plot(range(len(actual_in), len(actual_in) + len(actual_out)), actual_out,
                 label='actual_out')
        plt.plot(range(len(actual_in), len(actual_in) + len(actual_out)), actual_pred,
                 label='actual_out')
        plt.show()
        plt.savefig(fname='test')


class PIDThickenerPressureSimulation:

    def __init__(self,
                 set_value,
                 input_dim,
                 output_dim,
                 used_columns,
                 figs_path,
                 model,
                 random_seed=None,
                 device='cpu',
                 ):

        """
        :param model: 建模实验中保存的模型
        :param random_seed: 随机化种子，会决定浓密机的初始状态和
        """
        self.model = model
        self.device = device
        self.observation = [0.0 for x in range(output_dim)]
        self.observation_seq = None
        self.action = [0.0 for x in range(input_dim)]
        self.obs_name = used_columns[input_dim:]
        self.action_name = used_columns[:input_dim]
        self.set_value = set_value
        if random_seed is None:
            random_seed = random.randint(0, 1000000)
            print('random_seed:' + str(random_seed))
        self.random_seed = random_seed  # 随机种子作用
        self.simulation_time = 0  # 模拟仿真浓密机运行时间
        self.border = 0  # 绘图边界
        self.memory_state = None
        self.simulation_state_list = []
        self.figs_path = figs_path
        self.input_length = 60
        self.output_length = 30

        # 数据读取
        df_in = pd.concat([pd.read_csv('D:\\ZZX\\git\\SE-Control\\jupyter\\1-135-36005-unnormalized.csv'),
                           pd.read_csv('D:\\ZZX\\git\\SE-Control\\jupyter\\2-308-61861-unnormalized.csv')], axis=0,
                          ignore_index=True)
        self.df_scalar = DataFrameScalar()
        self.df_in_scale = self.df_scalar.transform(df_in,
                                                    stats_columns=["feed_c", "feed_f", "out_f", "out_c", "pressure"])

        self.df_in_scale = self.df_in_scale.drop(['time'], axis=1)
        self.Simulator = ThickenerSimulation(random_seed, model, arrayScalar(*self.df_scalar.get_stats()),
                                             self.input_length, self.output_length)

    def step(self, planning_action):
        """

        1. 利用model预测下一时刻的pressure, c_out
        2. 从dataset取出新的v_in, c_in
        3. 更新浓密机hidden_state
        4. 返回五元组

        Args:
            planning_action:

        Returns:

        """

        # 1 预测 output  test_in.shape = (input_chunk_length,1), test_cov.shape(input_chunk_length,4)
        print(self.simulation_time)

        # 此处planning_action为长度为1的list,且未归一化

        # 数据获取
        external_input = np.array([planning_action])
        covariable = self.df_in_scale[:self.input_length].drop(['out_f', 'fill_round'],
                                                               axis=1).values  # 取前三十个点的固定cov值（不包括output）

        # 读入上次预测的到的部分observation_seq
        if self.observation_seq is None:
            pass
        else:
            for i in range(self.output_length):
                covariable[self.input_length - self.output_length + i][1] = \
                    self.observation_seq.reshape(self.output_length, 1)[i][0]

        # numpy扩维转TimeSeries
        external_input = np.tile(external_input, (self.input_length, 1))

        pred_out_f = self.Simulator.simulate(external_input, covariable)

        # observation_seq最后一位作为observation并进行反归一化
        self.action = planning_action

        self.observation = pred_out_f[-1]

# TODO:绘图部分待测试
        # 绘图
        self.simulation_time = self.simulation_time + 1
        self.simulation_state_list.append(list(self.get_current_state().values()))

        if self.simulation_time % 10 == 0:  # 10周期绘图
            # 图1----仿真obs 和obs 设定值
            for pos in range(0, len(self.observation)):
                # plt.plot(range(self.border, self.simulation_time), [self.set_value[pos] for x in range(self.border, self.simulation_time)], label='set_value')
                plt.plot(range(self.border, self.simulation_time),
                         [self.simulation_state_list[x][0][pos] for x in range(self.border, self.simulation_time)],
                         label='planning')  # 仿真浓密机数据  -----暂时只选择pressure
                plt.title(self.obs_name[pos])
                plt.xlabel("time(minute)")
                plt.ylabel("control_value")
                plt.legend()
                try:
                    plt.savefig(
                        os.path.join(self.figs_path, f'simulation_{pos}_{self.border}.png')
                    )
                except Exception as e:
                    # import pdb
                    # pdb.set_trace()
                    raise e
                plt.close()

            # plt.plot(range(self.border, self.simulation_time),[self.set_value[pos] for x in range(self.border, self.simulation_time)], label='set_value')
            # test_df = pd.DataFrame(self.simulation_state_list[0])
            # test_df.to_csv(os.path.join(self.figs_path, "Simulation_list.csv"),index=False)

            # 画动作序列的图
            for pos in range(0, len(self.action)):
                # plt.plot(range(self.border, self.simulation_time), [self.set_value[pos] for x in range(self.border, self.simulation_time)], label='set_value')
                plt.plot(range(self.border, self.simulation_time),
                         [self.simulation_state_list[x][1][pos] for x in range(self.border, self.simulation_time)],
                         label='planning' + str(pos))  # 仿真浓密机数据  -----暂时只选择pressure
                plt.title(self.action_name[pos])
                plt.xlabel("time(minute)")
                plt.ylabel("control_value")
                plt.legend()
                try:
                    plt.savefig(
                        os.path.join(self.figs_path, 'action_' + str(pos) + '_' + str(self.border) + '_.png')
                    )
                except Exception as e:
                    # import pdb
                    # pdb.set_trace()
                    raise e
                plt.close()

            self.border = self.simulation_time
            # plt.plot(range(self.border, self.simulation_time),[self.set_value[pos] for x in range(self.border, self.simulation_time)], label='set_value')
            # test_df = pd.DataFrame(self.simulation_state_list[0])
            # test_df.to_csv(os.path.join(self.figs_path, "Simulation_list.csv"),index=False)

        # 4 返回五元组  (length, batch_size, 5)
        return self.get_current_state()

    def get_current_state(self):
        return {
            'observation': self.observation,
            'action': self.action,
        }
