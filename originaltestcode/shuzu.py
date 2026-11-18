import json

# 读取 JSON 文件
with open('/home/xuyuan/pan/xybag/scene0000_00.aggregation.json', 'r') as file:
    scene_data = json.load(file)

# 初始化一个空列表，用于存储所有的 segments
all_segments = []

# 遍历每个 segGroup，将它的 segments 添加到 all_segments 中
for seg_group in scene_data['segGroups']:
    all_segments.extend(seg_group['segments'])

# 将 allSegments 添加到原数据中
scene_data['allSegments'] = all_segments

# 输出更新后的 JSON 数据
with open('updated_scene_data.json', 'w') as file:
    json.dump(scene_data, file, indent=4)

# 打印更新后的数据
print(json.dumps(scene_data, indent=4))

