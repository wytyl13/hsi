import json
import os
import re
from urllib.parse import unquote

json_file = 'project-13-at-2026-05-23-06-43-aa7c4f4b.json' # 换成你的 JSON 名字

with open(json_file, 'r', encoding='utf-8') as f:
    tasks = json.load(f)

filename_map = {}
conflict_count = 0

print("🔍 开始侦查冲突文件...\n")
for task in tasks:
    # 只看有标签的任务
    if not task.get('annotations') or len(task['annotations']) == 0:
        continue
    
    image_url = task['data']['image']
    
    if '?d=' in image_url:
        original_path = unquote(image_url.split('?d=')[-1])
    else:
        original_path = unquote(image_url.split('/upload/')[-1])
        
    basename = os.path.basename(original_path)
    clean_filename = re.sub(r'^[a-f0-9]{8}-', '', basename)
    
    if clean_filename in filename_map:
        conflict_count += 1
        print(f"🚨 抓到第 {conflict_count} 个冲突文件名: {clean_filename}")
        print(f"   👉 占用者的 URL: {filename_map[clean_filename]}")
        print(f"   👉 新来者的 URL: {image_url}\n")
    else:
        filename_map[clean_filename] = image_url

print(f"==========================================")
print(f"侦查结束，共发现 {conflict_count} 次冲突。")