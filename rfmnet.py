import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.nn import init
import numpy as np

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    Modified from https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/SelfAttention.py
    '''

    def __init__(self, d_model, h, dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()

        self.d_model = d_model
        self.d_k = d_model // h
        self.d_v = d_model // h
        self.h = h

        self.fc_q = nn.Linear(d_model, h * self.d_k)
        self.fc_k = nn.Linear(d_model, h * self.d_k)
        self.fc_v = nn.Linear(d_model, h * self.d_v)
        self.fc_o = nn.Linear(h * self.d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, query_mask=None, key_mask=None, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_channel)
        :param keys: Keys (b_s, nk, d_channel)
        :param values: Values (b_s, nk, d_channel)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        if query_mask is not None:
            q = q * query_mask.view(b_s, nq, 1, 1).repeat(1, 1, self.h, 1).permute(0, 2, 1, 3)
        if key_mask is not None:
            k = k * key_mask.view(b_s, nk, 1, 1).repeat(1, 1, self.h, 1).permute(0, 2, 3, 1)
            v = v * key_mask.view(b_s, nk, 1, 1).repeat(1, 1, self.h, 1).permute(0, 2, 1, 3)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out, att

class OverlappedWithOneAtt(nn.Module):
    def __init__(self, channel, num_head=2, dropout=0.1):
        super(OverlappedWithOneAtt, self).__init__()
        self.dot_att = ScaledDotProductAttention(d_model=channel, h=num_head, dropout=dropout)
    
    def forward(self, q_x, kv_x):
        B, C, n_h, n_w, vq_h, vq_w = q_x.shape
        n_win = n_h * n_w
        seq_len_q = vq_h * vq_w
        # kv_x: (B, C, n_win, vk_h, vk_w)，每窗口是整张 sod 图，序列长度与 q 不同
        seq_len_kv = kv_x.shape[3] * kv_x.shape[4]

        # (B, C, n_h, n_w, vq_h, vq_w) -> (B, C, n_win, seq_len_q)
        q_x = q_x.reshape(B, C, n_win, seq_len_q)
        # (B, C, n_win, vk_h, vk_w) -> (B, C, n_win, seq_len_kv)
        kv_x = kv_x.reshape(B, C, n_win, seq_len_kv)

        # 多窗口拼成 batch；q 每窗口 seq_len_q，k/v 每窗口 seq_len_kv
        q = q_x.permute(0, 2, 3, 1).reshape(B * n_win, seq_len_q, C)
        k = kv_x.permute(0, 2, 3, 1).reshape(B * n_win, seq_len_kv, C)
        v = kv_x.permute(0, 2, 3, 1).reshape(B * n_win, seq_len_kv, C)

        att_out, att = self.dot_att(queries=q, keys=k, values=v)

        # att_out 每窗口长度为 seq_len_q -> (B, C, n_h, n_w, vq_h, vq_w)
        result_att = att_out.view(B, n_win, seq_len_q, C).permute(0, 3, 1, 2)
        result_att = result_att.reshape(B, C, n_h, n_w, vq_h, vq_w)

        return result_att
    


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class FusionBL(nn.Module):
    def __init__(self, input_channel, mg=False):
        super().__init__()
        self.conv_f = BasicConv2d(input_channel*2, input_channel, 3, padding=1)
        self.conv_fx = BasicConv2d(input_channel, input_channel, 3, padding=1)
        self.cls_pred_mg = nn.Sequential(
            BasicConv2d(input_channel, input_channel, kernel_size=3, padding=1),
            nn.Dropout2d(p=0.1), 
            nn.Conv2d(input_channel, 1, 1)
        )
        self.conv_3 = nn.Sequential(
            BasicConv2d(input_channel, input_channel, kernel_size=3, padding=1),
            BasicConv2d(input_channel, input_channel, kernel_size=3, padding=1),
            BasicConv2d(input_channel, input_channel, kernel_size=3, padding=1),
        )
        
        if mg==True:
            self.conv_g = BasicConv2d(input_channel, input_channel, 3, padding=1)
            self.conv_cat = BasicConv2d(input_channel*2, input_channel, 1)
            
        
    def forward(self, f_x, m_f=None, g_x=None, m_g=None):
        f_h, f_w = f_x.shape[2:]
        if not m_f == None:
            mf_h, mf_w = m_f.shape[2:]
            m_f = F.interpolate(m_f, size=(f_h, f_w), mode='bilinear', align_corners=True)
            f_x_m = f_x * m_f
            f_m_cat = torch.cat((f_x, f_x_m), dim=1)
            f_m_conv = self.conv_f(f_m_cat)
        else:
            mf_h, mf_w = f_h, f_w
            f_m_conv = self.conv_fx(f_x)
        
        if g_x==None and m_g==None:
            sg = F.interpolate(f_m_conv, size=(mf_h, mf_w), mode='bilinear', align_corners=True)
            mg_out = self.cls_pred_mg(sg)
            g_x_out = self.conv_3(f_m_conv)
        else:
            mg_up = F.interpolate(m_g, size=(f_h, f_w), mode='bilinear', align_corners=True)
            g_up = F.interpolate(g_x, size=(f_h, f_w), mode='bilinear', align_corners=True)
            g_mul = g_up * mg_up
            g_conv = self.conv_g(g_mul)
            fg_cat = torch.cat((f_m_conv, g_conv), dim=1)
            fg_cat_conv = self.conv_cat(fg_cat)
            g_x_out = self.conv_3(fg_cat_conv)
            sg = F.interpolate(g_x_out, size=(mf_h, mf_w), mode='bilinear', align_corners=True)
            mg_out=self.cls_pred_mg(sg)
        
        return [g_x_out, mg_out]

            
class RFMNet(nn.Module):
    def __init__(self, channel=256, imagenet_pretrained=True, shot=3, kernel_set=[8, 16, 32], step_set=[8, 16, 32], layers_set = [1, 1, 1]):
        super(RFMNet, self).__init__()
        
        self.shot = shot
        self.kernel_set = kernel_set
        self.step_set = step_set
        self.layers_set = layers_set
        self.swins = torchvision.models.swin_s(pretrained=imagenet_pretrained)
        self.swinssod = torchvision.models.swin_s(pretrained=imagenet_pretrained)
        
        self.x1_down_channel = BasicConv2d(96, channel, 1)
        self.x2_down_channel = BasicConv2d(192, channel, 1)
        self.x3_down_channel = BasicConv2d(384, channel, 1)
        self.x4_down_channel = BasicConv2d(768, channel, 1)
        
        self.sod_x1_down_channel = BasicConv2d(96, channel, 1)
        self.sod_x2_down_channel = BasicConv2d(192, channel, 1)
        self.sod_x3_down_channel = BasicConv2d(384, channel, 1)
        self.sod_x4_down_channel = BasicConv2d(768, channel, 1)
        
        self.shot_x2_down_channel = BasicConv2d(self.shot*channel, channel, 1)
        self.shot_x3_down_channel = BasicConv2d(self.shot*channel, channel, 1)
        self.shot_x4_down_channel = BasicConv2d(self.shot*channel, channel, 1)

        self.referred_overlapped_with_one_att_x2 = OverlappedWithOneAtt(channel=channel, num_head=4, dropout=0.3)
        self.alpha_refer_x2 = torch.tensor(0.5, requires_grad=True)
        self.conv_referred_x2 = BasicConv2d(channel, channel, 1)
        
        self.referred_overlapped_with_one_att_x3 = OverlappedWithOneAtt(channel=channel, num_head=4, dropout=0.3)
        self.alpha_refer_x3 = torch.tensor(0.5, requires_grad=True)
        self.conv_referred_x3 = BasicConv2d(channel, channel, 1)
        
        
        self.referred_overlapped_with_one_att_x4 = OverlappedWithOneAtt(channel=channel, num_head=4, dropout=0.3)
        self.alpha_refer_x4 = torch.tensor(0.5, requires_grad=True)
        self.conv_referred_x4 = BasicConv2d(channel, channel, 1)
        
        self.fbl_x43 = FusionBL(input_channel=channel,  mg=False)
        self.fbl_x32 = FusionBL(input_channel=channel,  mg=True)
        self.fbl_x21 = FusionBL(input_channel=channel,  mg=True)
        self.fbl_x11 = FusionBL(input_channel=channel,  mg=True)

    def sod_feature_extraction(self, sod_x):
        # SOD Feature Extraction
        sod_x = self.swinssod.features[0](sod_x)
        sod_x1 = self.swinssod.features[1](sod_x)
        
        sod_x2 = self.swinssod.features[2](sod_x1)
        sod_x2 = self.swinssod.features[3](sod_x2)
        
        sod_x3 = self.swinssod.features[4](sod_x2)
        sod_x3 = self.swinssod.features[5](sod_x3)
        
        sod_x4 = self.swinssod.features[6](sod_x3)
        sod_x4 = self.swinssod.features[7](sod_x4)
        
        sod_x1 = sod_x1.permute(0, 3, 1, 2)
        sod_x2 = sod_x2.permute(0, 3, 1, 2)
        sod_x3 = sod_x3.permute(0, 3, 1, 2)
        sod_x4 = sod_x4.permute(0, 3, 1, 2)
        
        # Down Channel
        sod_x1 = self.sod_x1_down_channel(sod_x1)    # bs, channel, H/4, W/4      # bs, channel, 128, 128
        sod_x2 = self.sod_x2_down_channel(sod_x2)    # bs, channel, H/8, W/8        # bs, channel, 64, 64
        sod_x3 = self.sod_x3_down_channel(sod_x3)    # bs, channel, H/16, W/16      # bs, channel, 32, 32
        sod_x4 = self.sod_x4_down_channel(sod_x4)    # bs, channel, H/32, W/32      # bs, channel, 16, 16
        
        return [sod_x1, sod_x2, sod_x3, sod_x4]

    def feature_extraction(self, x):
        # Feature Extraction
        x = self.swins.features[0](x)
        x1 = self.swins.features[1](x)
        
        x2 = self.swins.features[2](x1)
        x2 = self.swins.features[3](x2)
        
        x3 = self.swins.features[4](x2)
        x3 = self.swins.features[5](x3)
        
        x4 = self.swins.features[6](x3)
        x4 = self.swins.features[7](x4)
        
        x1 = x1.permute(0, 3, 1, 2)
        x2 = x2.permute(0, 3, 1, 2)
        x3 = x3.permute(0, 3, 1, 2)
        x4 = x4.permute(0, 3, 1, 2)
        
        # Down Channel
        x1 = self.x1_down_channel(x1)    # bs, channel, H/4, W/4      # bs, channel, 128, 128
        x2 = self.x2_down_channel(x2)    # bs, channel, H/8, W/8        # bs, channel, 64, 64
        x3 = self.x3_down_channel(x3)    # bs, channel, H/16, W/16      # bs, channel, 32, 32
        x4 = self.x4_down_channel(x4)    # bs, channel, H/32, W/32      # bs, channel, 16, 16
        
        return [x1, x2, x3, x4]
        
    def forward(self, x, sod_x='', salf_x='', sod_x_list='', ref_text_token = '', camo_text_token = ''):
        bs, _, H, W = x.shape
        kernel_set = self.kernel_set
        step_set = self.step_set

        x1, x2, x3, x4 = self.feature_extraction(x)
        sod_list = [self.sod_feature_extraction(sod_x_list[i]) for i in range(self.shot)]

        sod_x2_list = torch.cat([x[1] for x in sod_list], dim=1)
        sod_x2_down = self.shot_x2_down_channel(sod_x2_list)
        sod_x3_list = torch.cat([x[2] for x in sod_list], dim=1)
        sod_x3_down = self.shot_x3_down_channel(sod_x3_list)
        sod_x4_list = torch.cat([x[3] for x in sod_list], dim=1)
        sod_x4_down = self.shot_x2_down_channel(sod_x4_list)
        
        kernel_size_list = [int(kernel_set[0]), int(kernel_set[1]), int(kernel_set[2])]
        step_list = [int(step_set[0]), int(step_set[1]), int(step_set[2])]
        
        if self.layers_set[0] == 1:
            unfold_x2 = self.tensor_unfold(x2, kernel_size=kernel_size_list[0], step=step_list[0])
            sod_x2_kv = sod_x2_down.unsqueeze(2).repeat(1, 1, unfold_x2.shape[2]*unfold_x2.shape[3], 1, 1)
            ref_x2_atts = self.referred_overlapped_with_one_att_x2(q_x=unfold_x2, kv_x=sod_x2_kv)
            fold_ref_x2_atts = self.tensor_fold(ref_x2_atts, mode_x=x2, kernel_size=kernel_size_list[0], step=step_list[0])
            referred_x2 = self.alpha_refer_x2 * fold_ref_x2_atts + (1 - self.alpha_refer_x2) * x2
            referred_x2 = self.conv_referred_x2(referred_x2)
        else:
            referred_x2 = x2
        
        if self.layers_set[1] == 1:
            unfold_x3 = self.tensor_unfold(x3, kernel_size=kernel_size_list[1], step=step_list[1])
            sod_x3_kv = sod_x3_down.unsqueeze(2).repeat(1, 1, unfold_x3.shape[2]*unfold_x3.shape[3], 1, 1)
            ref_x3_atts = self.referred_overlapped_with_one_att_x3(q_x=unfold_x3, kv_x=sod_x3_kv)
            fold_ref_x3_atts = self.tensor_fold(ref_x3_atts, mode_x=x3, kernel_size=kernel_size_list[1], step=step_list[1])
            referred_x3 = self.alpha_refer_x3 * fold_ref_x3_atts + (1 - self.alpha_refer_x3) * x3
            referred_x3 = self.conv_referred_x3(referred_x3)
        else:
            referred_x3 = x3
        
        if self.layers_set[2] == 1:
            unfold_x4 = self.tensor_unfold(x4, kernel_size=kernel_size_list[2], step=step_list[2])
            sod_x4_kv = sod_x4_down.unsqueeze(2).repeat(1, 1, unfold_x4.shape[2]*unfold_x4.shape[3], 1, 1)
            ref_x4_atts = self.referred_overlapped_with_one_att_x4(q_x=unfold_x4, kv_x=sod_x4_kv)
            fold_ref_x4_atts = self.tensor_fold(ref_x4_atts, mode_x=x4, kernel_size=kernel_size_list[2], step=step_list[2])
            referred_x4 = self.alpha_refer_x4 * fold_ref_x4_atts + (1 - self.alpha_refer_x4) * x4
            referred_x4 = self.conv_referred_x4(referred_x4)
        else:
            referred_x4 = x4
        
        f_x2 = referred_x2
        f_x3 = referred_x3
        f_x4 = referred_x4

        g_out4, m_pred4 = self.fbl_x43(f_x=f_x4, m_f=None, g_x=None, m_g=None)
        m_pred4_out = F.interpolate(m_pred4, size=(H, W), mode='bilinear', align_corners=True)
        
        g_out3, m_pred3 = self.fbl_x32(f_x=f_x3, m_f=None, g_x=g_out4, m_g=m_pred4_out)
        m_pred3_out = F.interpolate(m_pred3, size=(H, W), mode='bilinear', align_corners=True)
        
        g_out2, m_pred2 = self.fbl_x21(f_x=f_x2, m_f=None, g_x=g_out3, m_g=m_pred3_out)
        m_pred2_out = F.interpolate(m_pred2, size=(H, W), mode='bilinear', align_corners=True)

        g_out1, m_pred1 = self.fbl_x11(f_x=x1, m_f=None, g_x=g_out2, m_g=m_pred2_out)
        m_pred1 = F.interpolate(m_pred1, size=(H, W), mode='bilinear', align_corners=True)

        
        return m_pred1, [m_pred2_out, m_pred3_out, m_pred4_out]
    
    def tensor_unfold(self, x, kernel_size=4, step=2):
        # x: [B, C, H, W]
        unfold_x = x.unfold(dimension=2, size=kernel_size, step=step).unfold(dimension=3, size=kernel_size, step=step) 
        # [b, c, h, w] --> [b, c, n_h, w, kernel_h] --> [b, c, n_h, n_w, kernel_h, kernel_w]
        return unfold_x
    
    def tensor_fold(self, x, mode_x, kernel_size=4, step=2):
        assert kernel_size / 2 <= step <= kernel_size
        B, C, n_h, n_w, k_h, k_w = x.shape
        H, W = mode_x.shape[2], mode_x.shape[3]
        if kernel_size == step and kernel_size == mode_x.shape[2]:
            return x.squeeze(2).squeeze(2)
        # x: (B, C, n_h, n_w, k_h, k_w) -> F.fold 要 (B, C*k_h*k_w, n_h*n_w)
        patches = x.permute(0, 1, 4, 5, 2, 3).contiguous().view(B, C * k_h * k_w, n_h * n_w)
        out_sum = F.fold(patches, output_size=(H, W), kernel_size=(k_h, k_w), stride=(step, step))
        ones_flat = torch.ones(B, C * k_h * k_w, n_h * n_w, dtype=x.dtype, device=x.device)
        count = F.fold(ones_flat, output_size=(H, W), kernel_size=(k_h, k_w), stride=(step, step))
        return out_sum / count.clamp(min=1.0)
