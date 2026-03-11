# 感知流（Topic 消息）
std_msgs/Header header          # timestamp + frame_id（tf2/rosbag 集成必须）
uint32 seq                      # 序号（检测丢失用）
string perception_type          # "visual", "tactile", "auditory", "proprioceptive", "procedural"
string description              # 人类可读描述（至少 5 字符）
string data                     # JSON 原始数据
string metadata                 # JSON 元数据
string session_id               # 关联 session（选填）
string collection               # 选填
