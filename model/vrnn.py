#!/usr/bin/python
# -*- coding:utf8 -*-
import numpy as np
import math
import os
import json

import torch
from torch import nn
from model.common import DBlock, PreProcess, MLP
from common import softplus, inverse_softplus, cov2logsigma, logsigma2cov, split_first_dim, merge_first_two_dims
from model.common import DiagMultivariateNormal as MultivariateNormal
from model.func import normal_differential_sample, multivariate_normal_kl_loss, zeros_like_with_shape

class VRNN(nn.Module):

    def __init__(self, input_size, state_size, observations_size, net_type='rnn', k=16, num_layers=1, D=5):

        super(VRNN, self).__init__()

        self.k = k
        self.D = D
        self.observations_size = observations_size
        self.state_size = state_size
        self.input_size = input_size
        self.num_layers = num_layers
        if net_type == 'rnn':
            RnnClass = torch.nn.RNN
        elif net_type == 'gru':
            RnnClass = torch.nn.GRU
        else:
            raise NotImplementedError('The posterior rnn with type %d is not implemented!' % net_type)

        self.rnn = RnnClass(3*k, k, num_layers)
        self.process_u = PreProcess(input_size, k)
        self.process_x = PreProcess(observations_size, k)
        self.process_z = PreProcess(state_size, k)

        self.posterior_gauss = DBlock(3*k, 3*k, state_size)
        self.prior_gauss = DBlock(k+k, 3*k, state_size)
        self.decoder = DBlock(2*k, 3*k, observations_size)

        # self.state_mu = None
        # self.state_logsigma = None
        # self.external_input_seq = None
        # self.observations_seq = None
        # self.initial_prior_mu, self.initial_prior_logsigma = None, None
        # self.sampled_state = None
        self.weight_initial_hidden_state = None
        self.h_seq = None
        self.external_input_seq_embed = None
        self.observations_seq_embed = None
        self.memory_state = None
        self.rnn_hidden_state_seq = None

    def forward_posterior(self, external_input_seq, observations_seq, memory_state=None):
        """
        ??????????????????????????????????????????????????????????????????????????????loss
        ?????????: ???????????????forward_prediction??????memory_state(h, rnn_hidden)
        Args:
            external_input_seq: ??????????????????(????????????????????????) (len, batch_size, input_size)
            observations_seq: ????????????(????????????) (len, batch_size, observations_size)
            memory_state: ????????????????????????????????????forward????????????????????????????????????memory_state???

        Returns:

        """
        external_input_seq_embed = self.process_u(external_input_seq)
        observations_seq_embed = self.process_x(observations_seq)

        l, batch_size, _ = external_input_seq.size()

        # ??????rnn???????????????????????????h?????????memory_state????????????memory_state??????
        h, rnn_hidden_state = (
            zeros_like_with_shape(external_input_seq, (batch_size, self.k)),
            zeros_like_with_shape(external_input_seq, (self.num_layers, batch_size, self.k))
        ) if memory_state is None else (memory_state['hn'], memory_state['rnn_hidden'])

        state_mu = []
        state_logsigma = []
        sampled_state = []
        h_seq = [h]
        rnn_hidden_state_seq = [rnn_hidden_state.transpose(1, 0)]
        for t in range(l):

            # ??????????????????t??????z???????????????
            z_t_mean, z_t_logsigma = self.posterior_gauss(
                torch.cat([observations_seq_embed[t], external_input_seq_embed[t], h], dim=-1)
            )
            # ????????????????????????z_t
            z_t = normal_differential_sample(
                MultivariateNormal(z_t_mean, logsigma2cov(z_t_logsigma))
            )

            #??????rnn????????????h
            output, rnn_hidden_state = self.rnn(torch.cat(
                [observations_seq_embed[t], external_input_seq_embed[t], self.process_z(z_t)], dim=-1
            ).unsqueeze(dim=0), rnn_hidden_state)
            h = output[0]

            # ?????????t?????????z????????????z????????????h???rnn?????????
            state_mu.append(z_t_mean)
            state_logsigma.append(z_t_logsigma)
            sampled_state.append(z_t)
            h_seq.append(h)
            rnn_hidden_state_seq.append(rnn_hidden_state.contiguous().transpose(1, 0))

        # ???list stack???????????????torch???tensor
        state_mu = torch.stack(state_mu, dim=0)
        state_logsigma = torch.stack(state_logsigma, dim=0)
        sampled_state = torch.stack(sampled_state, dim=0)
        h_seq = torch.stack(h_seq, dim=0)
        rnn_hidden_state_seq = torch.stack(rnn_hidden_state_seq, dim=0)

        outputs = {
            'state_mu': state_mu,
            'state_logsigma': state_logsigma,
            'sampled_state': sampled_state,
            'h_seq': h_seq,
            'external_input_seq_embed': external_input_seq_embed,
            'rnn_hidden_state_seq': rnn_hidden_state_seq
        }
        return outputs, {'hn': h, 'rnn_hidden': rnn_hidden_state}

    def forward_prediction(self, external_input_seq, max_prob=False, memory_state=None):
        """
        ??????????????????memory_state???????????????????????????(len,batch_size, input_size)????????????????????????????????????
        ?????????memory_state

        Args:
            external_input_seq:
            max_prob: ?????????????????????????????????????????????????????????max_prob???True???????????????????????????max_prob???False???????????????;
            memory_state:

        Returns:

        """

        l, batch_size, _ = external_input_seq.size()

        h, rnn_hidden_state = (
            zeros_like_with_shape(external_input_seq, (batch_size, self.k)),
            zeros_like_with_shape(external_input_seq, (self.num_layers, batch_size, self.k))
        ) if memory_state is None else (memory_state['hn'], memory_state['rnn_hidden'])

        external_input_seq_embed = self.process_u(external_input_seq)

        sampled_state = []
        h_seq = [h]
        rnn_hidden_state_seq = [rnn_hidden_state.transpose(1, 0)]

        for t in range(l):

            z_t_mean, z_t_logsigma = self.prior_gauss(
                torch.cat([external_input_seq_embed[t], h], dim=-1)
            )
            z_t = normal_differential_sample(
                MultivariateNormal(z_t_mean, logsigma2cov(z_t_logsigma))
            )
            z_t_embed = self.process_z(z_t)
            x_t_mean, x_t_logsigma = self.decoder(
                torch.cat([z_t_embed, h], dim=-1)
            )
            x_t = normal_differential_sample(
                MultivariateNormal(x_t_mean, logsigma2cov(x_t_logsigma))
            )
            output, rnn_hidden_state = self.rnn(torch.cat(
                [self.process_x(x_t), external_input_seq_embed[t], z_t_embed], dim=-1
            ).unsqueeze(dim=0), rnn_hidden_state)
            h = output[0]

            sampled_state.append(z_t)
            h_seq.append(h)
            rnn_hidden_state_seq.append(rnn_hidden_state.contiguous().transpose(1, 0))

        sampled_state = torch.stack(sampled_state, dim=0)
        h_seq = torch.stack(h_seq, dim=0)  # with shape (l+1, bs, k)
        rnn_hidden_state_seq = torch.stack(rnn_hidden_state_seq, dim=0)

        observations_dist = self.decode_observation(
            {'sampled_state': sampled_state,
             'h_seq': h_seq},
            mode='dist'
        )

        # ????????????????????????????????????lstm?????????
        if max_prob:
            observations_sample = observations_dist.loc
        else:
            observations_sample = normal_differential_sample(observations_dist)

        outputs = {
            'sampled_state': sampled_state,
            'h_seq': h_seq,
            'rnn_hidden_state_seq': rnn_hidden_state_seq,
            'predicted_dist': observations_dist,
            'predicted_seq': observations_sample
        }
        return outputs, {'hn': h, 'rnn_hidden': rnn_hidden_state}

    def call_loss(self, external_input_seq, observations_seq, memory_state=None):
        """
        ??????????????????????????????forward_posterior??????
        Returns:
        """
        outputs, memory_state = self.forward_posterior(external_input_seq, observations_seq, memory_state)

        h_seq = outputs['h_seq']
        rnn_hidden_state_seq = outputs['rnn_hidden_state_seq']
        state_mu = outputs['state_mu']
        state_logsigma = outputs['state_logsigma']
        external_input_seq_embed = outputs['external_input_seq_embed']

        l, batch_size, _ = observations_seq.shape

        D = self.D if self.training else 1

        kl_sum = 0

        predicted_h = h_seq[:-1]  # ??????h_seq??????????????????
        rnn_hidden_state_seq = rnn_hidden_state_seq[:-1]  # (length, bs, num_layers, k)

        # ?????????latent overshooting???d???over_shooting????????????????????????????????????????????????????????????????????????
        for d in range(D):
            length = predicted_h.size()[0]

            # ??????????????????????????????
            prior_z_t_seq_mean, prior_z_t_seq_logsigma = self.prior_gauss(
                torch.cat([external_input_seq_embed[-length:], predicted_h], dim=-1)
            )
            # ??????loss??????kl?????????
            kl_sum += multivariate_normal_kl_loss(
                state_mu[-length:],
                logsigma2cov(state_logsigma[-length:]),
                prior_z_t_seq_mean,
                logsigma2cov(prior_z_t_seq_logsigma)
            )

            # ??????reparameterization trick ??????z
            z_t_seq = normal_differential_sample(
                MultivariateNormal(prior_z_t_seq_mean, logsigma2cov(prior_z_t_seq_logsigma))
            )
            z_t_seq_embed = self.process_z(z_t_seq)
            x_t_seq_mean, x_t_seq_logsigma = self.decoder(
                torch.cat([z_t_seq_embed, predicted_h], dim=-1)
            )
            x_t_seq = normal_differential_sample(
                MultivariateNormal(x_t_seq_mean, logsigma2cov(x_t_seq_logsigma))
            )

            # region ?????????????????????????????????????????????predicted_h??????????????????????????????predicted_h???????????????
            # rnn_hidden_state 's shape : (num_layers, length*batch_size, k)
            output, rnn_hidden_state = self.rnn(
                merge_first_two_dims(
                    torch.cat(
                        [
                            self.process_x(x_t_seq),
                            external_input_seq_embed[-length:],
                            z_t_seq_embed
                        ], dim=-1
                    )
                ).unsqueeze(dim=0),
                merge_first_two_dims(rnn_hidden_state_seq).contiguous().transpose(1, 0)
            )
            rnn_hidden_state_seq = split_first_dim(rnn_hidden_state.contiguous().transpose(1, 0), (length, batch_size))[:-1]
            predicted_h = predicted_h[:-1]
            # endregion

        kl_sum = kl_sum/D

        # prior_z_t_seq_mean, prior_z_t_seq_logsigma = self.prior_gauss(
        #     torch.cat([self.external_input_seq_embed, self.h_seq[:-1]], dim=-1)
        # )
        #
        # kl_sum = multivariate_normal_kl_loss(
        #     self.state_mu,
        #     logsigma2cov(self.state_logsigma),
        #     prior_z_t_seq_mean,
        #     logsigma2cov(prior_z_t_seq_logsigma)
        # )
        # ???h????????????observation???generative??????
        observations_normal_dist = self.decode_observation(outputs, mode='dist')
        # ??????loss???????????????????????????????????????loss
        generative_likelihood = torch.sum(observations_normal_dist.log_prob(observations_seq))

        return {
            'loss': (kl_sum - generative_likelihood)/batch_size,
            'kl_loss': kl_sum/batch_size,
            'likelihood_loss': -generative_likelihood/batch_size
        }

    def decode_observation(self, outputs, mode='sample'):
        """

        Args:
            state: with shape (len, batch_size, state_size)
            mode: dist or sample

        Returns:

        """
        mean, logsigma = self.decoder(
            torch.cat([
                self.process_z(outputs['sampled_state']), outputs['h_seq'][:-1]
            ], dim=-1)
        )
        observations_normal_dist = MultivariateNormal(
            mean, logsigma2cov(logsigma)
        )
        if mode == 'dist':
            return observations_normal_dist
        elif mode == 'sample':
            return observations_normal_dist.sample()
