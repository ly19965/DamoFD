'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''

import os, sys
import re
import math
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch, argparse
from torch import nn
import torch.nn.functional as F
import PlainNet
import random
from PlainNet import parse_cmd_options, _create_netblock_list_from_str_, basic_blocks, super_blocks


def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_BN', action='store_true')
    parser.add_argument('--no_reslink', action='store_true')
    parser.add_argument('--use_se', action='store_true')
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt


class MasterNet(PlainNet.PlainNet):
    def __init__(self, argv=None, opt=None, num_classes=None, plainnet_struct=None, no_create=False,
                 no_reslink=None, no_BN=None, use_se=None):

        if argv is not None:
            module_opt = parse_cmd_options(argv)
        else:
            module_opt = None

        if no_BN is None:
            if module_opt is not None:
                no_BN = module_opt.no_BN
            else:
                no_BN = False

        if no_reslink is None:
            if module_opt is not None:
                no_reslink = module_opt.no_reslink
            else:
                no_reslink = False

        if use_se is None:
            if module_opt is not None:
                use_se = module_opt.use_se
            else:
                use_se = False


        super().__init__(argv=argv, opt=opt, num_classes=num_classes, plainnet_struct=plainnet_struct,
                                       no_create=no_create, no_reslink=no_reslink, no_BN=no_BN, use_se=use_se)
        self.last_channels = self.block_list[-1].out_channels
        self.fc_linear = basic_blocks.Linear(in_channels=self.last_channels, out_channels=self.num_classes, no_create=no_create)

        self.no_create = no_create
        self.no_reslink = no_reslink
        self.no_BN = no_BN
        self.use_se = use_se

        # bn eps
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eps = 1e-3


    def compute_sar_score(self, input_h):
        score_dict = {}
        cur_h = input_h
        d_0 = 3
        #alpha = 0.25
        alpha = 0.005
        log_eps = 0

        #self.block_list = ['SuperConvK3BNRELU(3,24,2,1)', 'SuperResK3K3(24,32,2,32,3)', 'SuperResK3K3(32,48,2,64,2)', 'SuperResK3K3(48,64,2,88,4)','SuperResK3K3(64,128,2,16,3)']
        for block_id, the_block in enumerate(self.block_list):
            sar_score = 0
            block_str = str(the_block)
            block_info = block_str[re.search('\(', block_str).span()[1]:-1].split(',')
            kernel_size_info = re.findall('K\d', block_str)
            len_layer = len(kernel_size_info)

            if len_layer == 1:
                kernel_size = int(kernel_size_info[0][-1])
            elif len_layer == 2:
                kernel_size = int(kernel_size_info[0][-1])
                kernel_size_1 = int(kernel_size_info[1][-1])
            else:
                kernel_size = int(kernel_size_info[1][-1])


            in_ch = int(block_info[0])
            out_ch = int(block_info[1])
            stride = int(block_info[2])
            sub_layer_num = int(block_info[-1])

            if re.search('IDWE', block_str):
                block_type = 'IDWE'
            elif len(kernel_size_info) == 2:
                block_type = 'ResKXKX'
            elif len(kernel_size_info) == 3:
                block_type = 'ResK1KXK1'
            else:
                block_type = 'Conv'

            item_1 = math.log(cur_h * cur_h * d_0)
            item_2 = 0
            item_3 = 0
            
			
            if block_type == 'Conv':
                for layer_idx in range(sub_layer_num):
                    item_2 += cur_h * cur_h * d_0 * math.log(in_ch / d_0 + log_eps)
                    in_ch = out_ch
                    if stride == 2:
                        cur_h /= 2
                        stride = 1
                    item_3 += kernel_size
            elif block_type == 'IDWE':
                bottleneck_channel = int(block_info[-2])
                expension_size = int(block_str[re.search('IDWE', block_str).span()[-1]])
                dw_channel = expension_size * bottleneck_channel
                for layer_idx in range(sub_layer_num):
                    item_2 += cur_h * cur_h * d_0 * math.log(in_ch / d_0 + log_eps)
                    item_2 += cur_h * cur_h * d_0 * math.log(dw_channel / d_0 + log_eps)  
                    if stride == 2:
                        cur_h /= 2
                        stride = 1
                    item_2 += cur_h * cur_h * d_0 * math.log(dw_channel / d_0 + log_eps) 
                    item_2 += cur_h * cur_h * d_0 * math.log(bottleneck_channel / d_0 + log_eps) 
                    item_2 += cur_h * cur_h * d_0 * math.log(dw_channel / d_0 + log_eps) 
                    item_2 += cur_h * cur_h * d_0 * math.log(dw_channel / d_0 + log_eps) 
                    in_ch = out_ch
                    item_3 += kernel_size * 2
                    item_3 += 4 * 1 # 4 1x1 conv
            elif block_type == 'ResK1KXK1':
                bottleneck_channel = int(block_info[-2])
                for layer_idx in range(sub_layer_num):
                    item_2 += cur_h * cur_h * d_0 * math.log(in_ch / d_0 + log_eps)
                    item_2 += cur_h * cur_h * d_0 * math.log(bottleneck_channel / d_0 + log_eps) 
                    if stride == 2:
                        cur_h /= 2
                        stride = 1
                    item_2 += cur_h * cur_h * d_0 * math.log(bottleneck_channel / d_0 + log_eps) 
                    item_2 += cur_h * cur_h * d_0 * math.log(out_ch / d_0) 
                    item_2 += cur_h * cur_h * d_0 * math.log(bottleneck_channel / d_0 + log_eps) 
                    item_2 += cur_h * cur_h * d_0 * math.log(bottleneck_channel / d_0 + log_eps) 
                    in_ch = out_ch
                    item_3 += kernel_size * 2
                    item_3 += 4 * 1 # 4 1x1 conv
            elif block_type == 'ResKXKX':
                bottleneck_channel = int(block_info[-2])
                #if block_str == 'SuperResK3K3(48,64,2,88,4)':
                #    import pdb;pdb.set_trace()
                for layer_idx in range(sub_layer_num):
                    item_2 += cur_h * cur_h * d_0 * math.log(in_ch / d_0 + log_eps)
                    if stride == 2:
                        cur_h /= 2
                        stride = 1
                    item_2 += cur_h * cur_h * d_0 * math.log(bottleneck_channel / d_0 + log_eps) 
                    in_ch = out_ch
                    item_3 += kernel_size
                    item_3 += kernel_size_1
                    #if block_str == 'SuperResK3K3(64,128,2,16,3)':
                    #    import pdb;pdb.set_trace()

            sar_score = math.log(item_1 + item_2 + log_eps) + alpha * item_3
            #print(item_1, item_2, item_3)
            score_dict['stage_{}'.format(block_id + 1)] = sar_score
        return score_dict


    def forward_ddsar_score(self, input_size):
        input_h = input_size[0]
        sar_score = self.compute_sar_score(input_h)
        s1_weight = 0.2
        s2_weight = 0.27
        s3_weight = 0.25
        s4_weight = 0.18
        s5_weight = 0.1 
        ddsar_score = 0
        ddsar_score += s1_weight * sar_score['stage_1']
        ddsar_score += s2_weight * sar_score['stage_2']
        ddsar_score += s3_weight * sar_score['stage_3']
        ddsar_score += s4_weight * sar_score['stage_4']
        ddsar_score += s5_weight * sar_score['stage_5']

        return ddsar_score

    def forward(self, x):
        output = x
        for block_id, the_block in enumerate(self.block_list):
            output = the_block(output)

        output = F.adaptive_avg_pool2d(output, output_size=1)

        output = torch.flatten(output, 1)
        output = self.fc_linear(output)
        return output

    def forward_pre_GAP(self, x):
        output = x
        for the_block in self.block_lisnfot:
            output = the_block(output)
        return output

    def get_FLOPs(self, input_resolution):
        the_res = input_resolution
        the_flops = 0
        for the_block in self.block_list:
            the_flops += the_block.get_FLOPs(the_res)
            the_res = the_block.get_output_resolution(the_res)

        #the_flops += self.fc_linear.get_FLOPs(the_res)

        return the_flops

    def get_model_size(self):
        the_size = 0
        for the_block in self.block_list:
            the_size += the_block.get_model_size()

        the_size += self.fc_linear.get_model_size()

        return the_size

    def get_num_layers(self):
        num_layers = 0
        for block in self.block_list:
            assert isinstance(block, super_blocks.PlainNetSuperBlockClass)
            num_layers += block.sub_layers
        return num_layers

    def replace_block(self, block_id, new_block):
        self.block_list[block_id] = new_block

        if block_id < len(self.block_list) - 1:
            if self.block_list[block_id + 1].in_channels != new_block.out_channels:
                self.block_list[block_id + 1].set_in_channels(new_block.out_channels)
        else:
            assert block_id == len(self.block_list) - 1
            self.last_channels = self.block_list[-1].out_channels
            if self.fc_linear.in_channels != self.last_channels:
                self.fc_linear.set_in_channels(self.last_channels)

        self.module_list = nn.ModuleList(self.block_list)

    def split(self, split_layer_threshold):
        new_str = ''
        for block in self.block_list:
            new_str += block.split(split_layer_threshold=split_layer_threshold)
        return new_str

    def init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=3.26033)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 3.26033 * np.sqrt(2 / (m.weight.shape[0] + m.weight.shape[1])))
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                pass

        for superblock in self.block_list:
            if not isinstance(superblock, super_blocks.PlainNetSuperBlockClass):
                continue
            for block in superblock.block_list:
                if not (isinstance(block, basic_blocks.ResBlock) or isinstance(block, basic_blocks.ResBlockProj)):
                    continue
                # print('---debug set bn weight zero in resblock {}:{}'.format(superblock, block))
                last_bn_block = None
                for inner_resblock in block.block_list:
                    if isinstance(inner_resblock, basic_blocks.BN):
                        last_bn_block = inner_resblock
                    pass
                pass  # end for
                assert last_bn_block is not None
                # print('-------- last_bn_block={}'.format(last_bn_block))
                nn.init.zeros_(last_bn_block.netblock.weight)
