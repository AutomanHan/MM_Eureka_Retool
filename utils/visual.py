import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.patches import FancyBboxPatch

def visualize_text_with_probability(texts, probabilities, colormap='Blues', 
                                  figsize=(12, 6), save_path=None):
    """
    可视化显示文本及其对应的概率，用颜色深浅表示概率高低
    
    参数:
    texts: list of str, 要显示的文本列表
    probabilities: list of float, 对应的概率列表
    colormap: str, 颜色映射名称
    figsize: tuple, 图形尺寸
    save_path: str, 保存路径，如果为None则不保存
    """
    
    # 验证输入
    if len(texts) != len(probabilities):
        raise ValueError("文本列表和概率列表长度必须相同")
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    # 创建颜色映射
    cmap = plt.get_cmap(colormap)
    # norm = mcolors.Normalize(vmin=min(probabilities), vmax=max(probabilities))
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    
    # 设置初始位置
    x_pos = 0
    y_pos = 0.5
    max_width = 10  # 每行最大宽度
    
    # 为每个文本创建带颜色的框
    for text, prob in zip(texts, probabilities):
        # 获取颜色
        color = cmap(norm(prob))
        
        # 估算文本宽度
        text_width = len(text) * 0.15
        
        # 如果当前行放不下，换行
        if x_pos + text_width > max_width:
            x_pos = 0
            y_pos -= 0.15
        
        # 创建带圆角的文本框
        bbox = FancyBboxPatch(
            (x_pos, y_pos), text_width, 0.1,
            boxstyle="round,pad=0.02",
            facecolor=color,
            edgecolor='black',
            linewidth=1
        )
        
        ax.add_patch(bbox)
        
        # 添加文本
        ax.text(
            x_pos + text_width / 2, y_pos + 0.05,
            text,
            ha='center', va='center',
            fontsize=10,
            weight='bold',
            color='white' if prob > (max(probabilities) + min(probabilities)) / 2 else 'black'
        )
        
        # 在文本下方显示概率值
        ax.text(
            x_pos + text_width / 2, y_pos - 0.02,
            f'{prob:.3f}',
            ha='center', va='top',
            fontsize=8,
            color='gray'
        )
        
        x_pos += text_width + 0.1
    
    # 设置图形属性
    ax.set_xlim(0, max_width)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('Probability', rotation=270, labelpad=15)
    
    plt.tight_layout()
    
    # 保存图片
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图片已保存到: {save_path}")
    
    plt.close()  # 关闭图形，释放内存

# 示例用法
if __name__ == "__main__":
    # 示例数据
    example_texts = ["The", "quick", "brown", "fox", "jumps", "over", "the", "lazy", "dog"]
    example_probs = [1.0, 0.8, 0.3, 0.4, 0.2, 0.5, 0.7, 0.25, 0.35]
    
    # 保存为图片文件
    visualize_text_with_probability(example_texts, example_probs, 
                                  save_path='./visual_jpg/text_probability_visualization.png')
    
    # 中文示例
    chinese_texts = ["今天", "天气", "很好", "我们", "去", "公园", "散步"]
    chinese_probs = [0.9, 0.7, 0.6, 0.8, 0.5, 0.4, 0.3]
    
    visualize_text_with_probability(chinese_texts, chinese_probs,
                                  save_path='./visual_jpg/chinese_text_probability.png')