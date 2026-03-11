# 单条记忆（recall 返回）
int64 id
string content
string type                     # "fact" / "perception"
string perception_type          # 空 = fact
float64 confidence
float64 rrf_score               # SDK 返回 _rrf_score，node 去下划线
string context_json             # SDK 返回 context，node 重命名
string session_id
string created_at
