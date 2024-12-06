import torch
import comfy.ops
from comfy.ldm.flux.redux import ReduxImageEncoder
from comfy.text_encoders.flux import FluxClipModel, FluxTokenizer
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
            "clip": ("CLIP",),
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
        # 使用 Flux 的双编码器系统
        self.flux_clip = FluxClipModel(device="cuda", dtype=torch.float16)
        
        # 特征解耦头
        self.style_projectors = torch.nn.ModuleDict({
            'style': ops.Linear(4096, 4096),
            'color': ops.Linear(4096, 4096),
            'content': ops.Linear(4096, 4096),
            'structure': ops.Linear(4096, 4096),
            'texture': ops.Linear(4096, 4096)
        })

    def get_style_tokens(self, clip):
        """生成风格相关的 token"""
        style_prompts = {
            "l": [  # CLIP-L prompts
                "artistic style elements",
                "color and lighting",
                "main subject and content",
                "composition layout",
                "surface details"
            ],
            "t5xxl": [  # T5XXL prompts
                "detailed artistic style analysis",
                "comprehensive color scheme",
                "content semantic meaning",
                "structural composition",
                "texture patterns"
            ]
        }
        # 使用新的 tokenization 方式
        tokens = clip.tokenize(style_prompts["l"])
        tokens["t5xxl"] = clip.tokenize(style_prompts["t5xxl"])["t5xxl"]
        return tokens

    def decouple_features(self, image_features, clip):
        """使用 Flux 双编码器进行特征解耦"""
        # 获取风格 tokens
        style_tokens = self.get_style_tokens(clip)
        
        # 使用 Flux 的双编码器获取特征
        t5_features, clip_l_features = self.flux_clip.encode_token_weights(style_tokens)
        
        # 解耦特征字典
        decoupled = {}
        
        # 对每个风格维度进行处理
        for idx, (name, projector) in enumerate(self.style_projectors.items()):
            # 结合 T5 和 CLIP-L 的特征，并扩展维度以匹配 batch_size
            t5_proj = t5_features[idx].unsqueeze(0).expand(image_features.shape[0], -1)  # [batch_size, 4096]
            clip_l_proj = clip_l_features[idx].unsqueeze(0).expand(image_features.shape[0], -1)  # [batch_size, 4096]
            
            # 投影图像特征
            img_proj = projector(image_features)  # [batch_size, 4096]
            
            # 融合三种特征，保持 batch 维度
            combined = (t5_proj + clip_l_proj + img_proj) / 3.0  # [batch_size, 4096]
            decoupled[name] = combined
            
        return decoupled

    def apply_style(self, clip_vision_output, style_model, conditioning, clip,
                   style_weight=1.0, color_weight=1.0, content_weight=1.0,
                   structure_weight=1.0, texture_weight=1.0,
                   similarity_threshold=0.7, enhancement_base=1.5):
        
        # 获取图像特征
        image_cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1)
        
        # 应用特征解耦
        decoupled_features = self.decouple_features(image_cond, clip)
        
        # 应用权重
        weights = {
            'style': style_weight,
            'color': color_weight,
            'content': content_weight,
            'structure': structure_weight,
            'texture': texture_weight
        }
        
        # 合并解耦特征
        combined_features = torch.zeros_like(image_cond)
        for name, features in decoupled_features.items():
            # 应用权重和增强
            enhanced_features = features * weights[name] * enhancement_base
            combined_features += enhanced_features
        
        # 归一化
        combined_features = combined_features / len(weights)
        
        # 构建新的条件
        c = []
        for t in conditioning:
            n = [torch.cat((t[0], combined_features.unsqueeze(0)), dim=1), t[1].copy()]
            c.append(n)
            
        return (c,)

