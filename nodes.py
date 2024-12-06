import torch
import comfy.ops
from comfy.ldm.flux.redux import ReduxImageEncoder
import math

# 获取ops引用
ops = comfy.ops.manual_cast

class StyleModelAdvancedApply:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "conditioning": ("CONDITIONING",),
            "style_model": ("STYLE_MODEL",),
            "clip_vision_output": ("CLIP_VISION_OUTPUT",),
            "style_weight": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 10.0,
                "step": 0.01,
                "tooltip": "控制整体艺术风格的权重"
            }),
            "color_weight": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 10.0,
                "step": 0.01,
                "tooltip": "控制颜色特征的权重"
            }),
            "content_weight": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 10.0,
                "step": 0.01,
                "tooltip": "控制内容语义的权重"
            }),
            "structure_weight": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 10.0,
                "step": 0.01,
                "tooltip": "控制结构布局的权重"
            }),
            "texture_weight": ("FLOAT", {
                "default": 1.0,
                "min": 0.0,
                "max": 10.0,
                "step": 0.01,
                "tooltip": "控制纹理细节的权重"
            }),
            "similarity_threshold": ("FLOAT", {
                "default": 0.7,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "tooltip": "特征相似度阈值，超过此值的区域将被替换"
            }),
            "enhancement_base": ("FLOAT", {
                "default": 1.5,
                "min": 1.0,
                "max": 3.0,
                "step": 0.1,
                "tooltip": "文本特征替换的基础增强系数"
            })
        }}
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_style"
    CATEGORY = "conditioning/style_model"

    def __init__(self):
        self.text_projector = ops.Linear(4096, 4096)  # 保持维度一致
        # 为不同类型特征设置增强系数
        self.enhancement_factors = {
            'style': 1.2,    # 风格特征增强系数
            'color': 1.0,    # 颜色特征增强系数
            'content': 1.1,  # 内容特征增强系数
            'structure': 1.3, # 结构特征增强系数
            'texture': 1.0   # 纹理特征增强系数
        }

    def compute_similarity(self, text_feat, image_feat):
        """计算多种相似度的组合"""
        # 1. 余弦相似度
        cos_sim = torch.cosine_similarity(text_feat, image_feat, dim=-1)
        
        # 2. L2距离相似度（归一化后的欧氏距离）
        l2_dist = torch.norm(text_feat - image_feat, p=2, dim=-1)
        l2_sim = 1 / (1 + l2_dist)  # 转换为相似度
        
        # 3. 点积相似度（考虑特征的强度）
        dot_sim = torch.sum(text_feat * image_feat, dim=-1)
        dot_sim = torch.tanh(dot_sim)  # 归一化到[-1,1]
        
        # 4. 注意力相似度
        attn_weights = torch.softmax(torch.matmul(text_feat, image_feat.transpose(-2, -1)) / math.sqrt(text_feat.size(-1)), dim=-1)
        attn_sim = torch.mean(attn_weights, dim=-1)
        
        # 组合所有相似度（可以调整权重）
        combined_sim = (
            0.4 * cos_sim +
            0.2 * l2_sim +
            0.2 * dot_sim +
            0.2 * attn_sim
        )
        
        return combined_sim.mean()

    def apply_style(self, clip_vision_output, style_model, conditioning,
                   style_weight=1.0, color_weight=1.0, content_weight=1.0,
                   structure_weight=1.0, texture_weight=1.0,
                   similarity_threshold=0.7, enhancement_base=1.5):
        
        # 获取图像特征并展平
        image_cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1)
        
        # 获取文本特征并调整维度
        text_features = conditioning[0][0]  # [batch_size, seq_len, 4096]
        text_features = text_features.mean(dim=1)  # [batch_size, 4096]
        
        # 投影文本特征（保持4096维度）
        text_features = self.text_projector(text_features)  # [batch_size, 4096]
        
        # 确保batch维度匹配
        if text_features.shape[0] != image_cond.shape[0]:
            text_features = text_features.expand(image_cond.shape[0], -1)
        
        # 分解图像特征为5个区域
        feature_size = image_cond.shape[-1]  # 4096
        splits = feature_size // 5  # 每部分约819维
        
        # 分离图像的不同类型特征
        image_features = {
            'style': image_cond[..., :splits],
            'color': image_cond[..., splits:splits*2],
            'content': image_cond[..., splits*2:splits*3],
            'structure': image_cond[..., splits*3:splits*4],
            'texture': image_cond[..., splits*4:]
        }
        
        # 计算每个区域与文本特征的相似度
        similarities = {}
        for key, region_features in image_features.items():
            # 将文本特征调整为对应区域的维度
            region_text_features = text_features[..., :region_features.shape[-1]]
            # 使用多种相似度度量的组合
            similarities[key] = self.compute_similarity(region_text_features, region_features)
        
        # 根据相似度和阈值决定替换，并应用增强
        final_features = {}
        weights = {
            'style': style_weight,
            'color': color_weight,
            'content': content_weight,
            'structure': structure_weight,
            'texture': texture_weight
        }
        
        for key in image_features:
            if similarities[key] > similarity_threshold:
                # 相似度高的区域，用增强后的文本特征替换
                region_size = image_features[key].shape[-1]
                # 计算动态增强系数
                dynamic_factor = enhancement_base * self.enhancement_factors[key]
                # 应用特征替换和增强
                final_features[key] = text_features[..., :region_size] * weights[key] * dynamic_factor
            else:
                # 保持原图像特征
                final_features[key] = image_features[key] * weights[key]
        
        # 合并所有特征
        combined_cond = torch.cat([
            final_features['style'],
            final_features['color'],
            final_features['content'],
            final_features['structure'],
            final_features['texture']
        ], dim=-1).unsqueeze(dim=0)
        
        # 构建新的条件
        c = []
        for t in conditioning:
            n = [torch.cat((t[0], combined_cond), dim=1), t[1].copy()]
            c.append(n)
            
        return (c,)

