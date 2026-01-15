"""
後処理モジュール - YOLO出力のパース
"""
from dataclasses import dataclass
import torch
from vision_msgs.msg import (
    Detection2D,
    Detection2DArray,
    BoundingBox2D,
    ObjectHypothesisWithPose,
    Pose2D,  # vision_msgs版のPose2D（BoundingBox2D用）
    Point2D,  # vision_msgs版のPoint2D
)
from geometry_msgs.msg import PoseWithCovariance, Pose, Point, Quaternion
from std_msgs.msg import Header
from builtin_interfaces.msg import Time


# カスタムモデルのクラス名（3クラス）
CLASS_NAMES = ["1", "2", "3"]

# 検出結果のスコア閾値
SCORE_THRESHOLD = 0.5


@dataclass
class Detection:
    """検出結果"""
    x_center: float
    y_center: float
    width: float
    height: float
    class_id: int
    class_name: str
    score: float


def parse_yolo_output(
    output: torch.Tensor,
    timestamp: Time,
    frame_id: str
) -> Detection2DArray | None:
    """
    YOLOの出力をパースしてDetection2DArrayに変換
    
    Args:
        output: モデル出力 (1, num_classes + 4, num_predictions)
                このモデルの場合: (1, 7, num_predictions) - 3クラス + 4 bbox座標
        timestamp: タイムスタンプ
        frame_id: フレームID
    
    Returns:
        Detection2DArray または None（有効な検出がない場合）
    """
    # Tensorをnumpy配列に変換
    output = output.squeeze(0)  # (7, num_predictions)
    
    if isinstance(output, torch.Tensor):
        output = output.cpu().numpy()
    
    num_features, num_predictions = output.shape
    num_classes = num_features - 4  # 4はbbox座標
    
    # 有効なpredictionを収集
    valid_predictions: list[tuple[int, list[tuple[int, float]]]] = []
    
    for i in range(num_predictions):
        # スコアが閾値以上のクラスを収集
        valid_classes = []
        for class_id in range(num_classes):
            score = float(output[4 + class_id, i])
            if score >= SCORE_THRESHOLD:
                valid_classes.append((class_id, score))
        
        # スコアが閾値以上のクラスが1つ以上ある場合のみ有効
        if valid_classes:
            valid_predictions.append((i, valid_classes))
    
    # 有効な検出結果が1つもない場合はNoneを返す
    if not valid_predictions:
        print("✅ 推論完了: []")
        return None
    
    # 検出結果をログ出力
    log_entries = []
    for i, valid_classes in valid_predictions:
        x_center = output[0, i]
        y_center = output[1, i]
        # 最もスコアが高いクラスを取得
        best_class_id, _ = max(valid_classes, key=lambda x: x[1])
        log_entries.append(f"({CLASS_NAMES[best_class_id]}, ({x_center:.1f}, {y_center:.1f}))")
    
    separator = ',\n  '
    print(f"✅ 推論完了: [\n  {separator.join(log_entries)}\n]")
    
    # Detection2DArrayを構築
    header = Header()
    header.stamp = timestamp
    header.frame_id = frame_id
    
    detections = []
    
    for i, valid_classes in valid_predictions:
        # bbox座標を取得
        x_center = float(output[0, i])
        y_center = float(output[1, i])
        width = float(output[2, i])
        height = float(output[3, i])
        
        # BoundingBox2Dを作成
        bbox = BoundingBox2D()
        bbox.center = Pose2D()
        bbox.center.position = Point2D()
        bbox.center.position.x = x_center
        bbox.center.position.y = y_center
        bbox.center.theta = 0.0
        bbox.size_x = width
        bbox.size_y = height
        
        # ObjectHypothesisWithPoseのリストを作成
        results = []
        for class_id, score in valid_classes:
            hypothesis = ObjectHypothesisWithPose()
            hypothesis.hypothesis.class_id = CLASS_NAMES[class_id]
            hypothesis.hypothesis.score = score
            hypothesis.pose = PoseWithCovariance()
            hypothesis.pose.pose = Pose()
            hypothesis.pose.pose.position = Point(x=0.0, y=0.0, z=0.0)
            hypothesis.pose.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=0.0)
            hypothesis.pose.covariance = [0.0] * 36
            results.append(hypothesis)
        
        # Detection2Dを作成
        detection = Detection2D()
        detection.header = Header()
        detection.header.stamp = timestamp
        detection.header.frame_id = frame_id
        detection.results = results
        detection.bbox = bbox
        detection.id = ""
        
        detections.append(detection)
    
    # Detection2DArrayを作成
    detection_array = Detection2DArray()
    detection_array.header = header
    detection_array.detections = detections
    
    return detection_array
