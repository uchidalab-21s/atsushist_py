"""
画像前処理モジュール - Ultralyticsスタイルの前処理
"""
import numpy as np
import torch
from PIL import Image


def letterbox(
    img: np.ndarray,
    target_size: tuple[int, int] = (640, 640),
    fill_color: tuple[int, int, int] = (114, 114, 114)
) -> np.ndarray:
    """
    Letterbox処理: アスペクト比を保持してリサイズし、パディングを追加
    
    Args:
        img: 入力画像 (H, W, C) RGB形式
        target_size: ターゲットサイズ (width, height)
        fill_color: パディング色 (R, G, B)
    
    Returns:
        letterbox処理された画像 (H, W, C)
    """
    target_w, target_h = target_size
    img_h, img_w = img.shape[:2]
    
    # スケール比を計算（アスペクト比を保持）
    scale = min(target_w / img_w, target_h / img_h)
    new_w = int(img_w * scale)
    new_h = int(img_h * scale)
    
    # PILでリサイズ（古いPillowバージョンとの互換性）
    pil_img = Image.fromarray(img)
    try:
        resample_method = Image.Resampling.BILINEAR
    except AttributeError:
        # Pillow < 9.1.0
        resample_method = Image.BILINEAR
    resized = pil_img.resize((new_w, new_h), resample_method)
    resized = np.array(resized)
    
    # パディングを計算
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    
    # 新しい画像を作成（パディング色で塗りつぶし）
    result = np.full((target_h, target_w, 3), fill_color, dtype=np.uint8)
    
    # リサイズした画像を中央に配置
    result[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    
    return result


def preprocess_image(
    img: np.ndarray,
    target_size: tuple[int, int] = (640, 640),
    device: torch.device = None
) -> torch.Tensor:
    """
    画像の前処理を実行
    
    Args:
        img: 入力画像 (H, W, C) RGB形式、uint8
        target_size: ターゲットサイズ (width, height)
        device: PyTorchデバイス
    
    Returns:
        前処理済みテンソル (1, 3, H, W)、値は0.0〜1.0
    """
    if device is None:
        device = torch.device('cpu')
    
    # 1. Letterbox処理
    letterboxed = letterbox(img, target_size, fill_color=(114, 114, 114))
    
    # 2. HWC -> CHW に変換し、正規化
    # (H, W, C) -> (C, H, W)
    tensor = torch.from_numpy(letterboxed).permute(2, 0, 1).float()
    tensor = tensor / 255.0  # 0.0〜1.0に正規化
    
    # 3. バッチ次元を追加 (1, C, H, W)
    tensor = tensor.unsqueeze(0)
    
    return tensor.to(device)
