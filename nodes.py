import torch
import comfy.ops
from comfy.ldm.flux.redux import ReduxImageEncoder

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
            })
        }}
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "apply_style"
    CATEGORY = "conditioning/style_model"

    def apply_style(self, clip_vision_output, style_model, conditioning,
                   style_weight=1.0, color_weight=1.0, content_weight=1.0,
                   structure_weight=1.0, texture_weight=1.0):
        
        # 获取原始特征并展平
        cond = style_model.get_cond(clip_vision_output).flatten(start_dim=0, end_dim=1)
        
        # 分解特征 - 基于cond的维度
        feature_size = cond.shape[-1]
        splits = feature_size // 5  # 将特征平均分成5份
        
        # 分离不同类型的特征
        features = {
            'style': cond[..., :splits],
            'color': cond[..., splits:splits*2],
            'content': cond[..., splits*2:splits*3],
            'structure': cond[..., splits*3:splits*4],
            'texture': cond[..., splits*4:]
        }
        
        # 应用权重
        weighted_features = {
            'style': features['style'] * style_weight,
            'color': features['color'] * color_weight,
            'content': features['content'] * content_weight,
            'structure': features['structure'] * structure_weight,
            'texture': features['texture'] * texture_weight
        }
        
        # 合并加权特征
        combined_cond = torch.cat([
            weighted_features['style'],
            weighted_features['color'],
            weighted_features['content'],
            weighted_features['structure'],
            weighted_features['texture']
        ], dim=-1).unsqueeze(dim=0)
        
        # 构建新的条件
        c = []
        for t in conditioning:
            n = [torch.cat((t[0], combined_cond), dim=1), t[1].copy()]
            c.append(n)
            
        return (c,)