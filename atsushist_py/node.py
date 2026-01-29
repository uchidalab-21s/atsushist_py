#!/usr/bin/env python3
"""
atsushist_py - YOLO物体検出ROS2ノード
PyTorchを使用した画像処理とONNXRuntimeでの推論
"""
import os
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, CompressedImage
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge
from builtin_interfaces.msg import Time

from atsushist_py.preprocessing import preprocess_image
from atsushist_py.postprocessing import parse_yolo_output


class AtsushistNode(Node):
    """YOLO物体検出ノード"""
    
    def __init__(self):
        super().__init__('atsushist_node')
        
        # パラメータ
        self.declare_parameter('model_path', '')
        self.declare_parameter('image_size', 640)
        self.declare_parameter('device', 'cpu')
        
        model_path_param = self.get_parameter('model_path').value
        self.image_size = self.get_parameter('image_size').value
        device_param = self.get_parameter('device').value
        
        # デバイス設定
        self.device = torch.device(device_param if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f'🔧 デバイス: {self.device}')
        
        # モデルパスの解決
        if model_path_param:
            model_path = Path(model_path_param)
        else:
            # デフォルト: パッケージ内のモデルを使用
            # まずインストールされたモデルを探す
            from ament_index_python.packages import get_package_share_directory
            try:
                share_dir = get_package_share_directory('atsushist_py')
                model_path = Path(share_dir) / 'model' / 'atsushist.onnx'
            except Exception:
                # 開発時はソースディレクトリを使用
                model_path = Path(__file__).parent.parent / 'model' / 'atsushist.onnx'
        
        if not model_path.exists():
            self.get_logger().error(f'❌ モデルファイルが見つかりません: {model_path}')
            raise FileNotFoundError(f'Model not found: {model_path}')
        
        self.get_logger().info(f'📦 モデル読み込み中: {model_path}')
        
        # ONNXRuntimeセッションの作成
        # PyTorchのテンソル操作との互換性のためExecutionProviderを設定
        providers = ['CPUExecutionProvider']
        if device_param == 'cuda' and 'CUDAExecutionProvider' in ort.get_available_providers():
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.get_logger().info(f'✅ モデル読み込み完了')
        
        # CV Bridge
        self.bridge = CvBridge()
        
        # QoS設定
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # サブスクライバー
        self.subscription = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.image_callback,
            qos
        )
        
        # パブリッシャー
        self.img_publisher = self.create_publisher(CompressedImage, '/images', 10)
        self.det_publisher = self.create_publisher(Detection2DArray, '/detections', 10)
        
        self.get_logger().info('🚀 atsushist_node 起動完了')
    
    def image_callback(self, msg: CompressedImage):
        """画像コールバック"""
        self.get_logger().info('📷 画像を受信')
        
        try:
            # 圧縮画像をnumpy配列に変換（BGRで取得してRGBに変換）
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='rgb8')
            
            # 画像サイズ設定
            target_size = (self.image_size, self.image_size)
            
            # 前処理（PyTorchを使用）
            input_tensor = preprocess_image(cv_image, target_size, self.device)
            
            # ONNXRuntimeで推論
            self.get_logger().info('🔮 推論を実行中...')
            input_numpy = input_tensor.cpu().numpy()
            outputs = self.session.run(None, {self.input_name: input_numpy})
            
            # 出力をPyTorchテンソルに変換
            output_tensor = torch.from_numpy(outputs[0])
            
            # タイムスタンプを取得
            timestamp = self.get_clock().now().to_msg()
            
            # 画像メッセージのtimestampを書き換えてpublish
            msg.header.stamp = timestamp
            self.img_publisher.publish(msg)
            
            # 後処理
            detections = parse_yolo_output(
                output_tensor,
                timestamp,
                msg.header.frame_id
            )
            
            # 有効な検出結果がある場合のみpublish
            if detections is not None:
                self.det_publisher.publish(detections)
            
        except Exception as e:
            self.get_logger().error(f'❌ 処理エラー: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())


def main(args=None):
    """メイン関数"""
    rclpy.init(args=args)
    
    node = AtsushistNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
