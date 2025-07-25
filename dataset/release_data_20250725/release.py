import json
import sys

path = sys.argv[1]
save_path = f"gemini_2.5_pro_judge_{path.split('/')[-2]}.jsonl"

count = 0
with open(save_path, 'w') as w:
    with open(path, 'r') as f:
        for index, line in enumerate(f):
            data = json.loads(line)
            result = {
                "question": data['question'],
                "query_class_ans": data.get('query_class_ans', ''),
                "checklist": data['checklist'],
                "class": data["class"],
                "model_infer_result": data["model_infer_result"],
                #"now_infer_model": data["now_infer_model"],
                "gemini_reason": data["gemini_reason"],
                "gemini_mllm_score": data["gemini_ans"],
            }
            json_data = json.dumps(result, ensure_ascii=False)
            w.write(json_data + '\n')
            count += 1

print(f"total: {count} | save_path: {save_path}")
#print(data)