#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ArtifactsBench数据处理脚本
读取artifacts_bench.jsonl文件，调用多个LLM模型获取回复

输出数据结构说明：
每条记录包含以下字段：
- index: 原始数据索引
- question: 原始问题
- model_name: 使用的模型名称
- response: 模型回复
- input_tokens: 输入token数量
- output_tokens: 输出token数量
- success: 是否成功
- processed_at: 处理时间
"""

import json
import os
import sys
import time
import random
import logging
from datetime import datetime
from http import HTTPStatus
from openai import OpenAI, RateLimitError, InternalServerError, APITimeoutError
import ast
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

def setup_logging(log_file):
    """设置日志记录"""
    logging.basicConfig(
        filename=log_file,
        level=logging.ERROR,
        format='%(asctime)s-%(levelname)s-%(message)s'
    ) 

def save_json(save_data, save_path):
    """保存JSON数据到文件"""
    # 确保目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 写入JSON文件
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

def read_json(path):
    """读取JSON文件"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def read_jsonl(path):
    """读取JSONL文件"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    # 尝试解析为JSON
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError:
                    # 如果JSON解析失败，尝试用ast.literal_eval解析Python字典格式
                    try:
                        item = ast.literal_eval(line)
                        data.append(item)
                    except:
                        print(f"无法解析行: {line}")
                        continue
    return data

def save_jsonl(data, path):
    """保存JSONL文件"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def load_existing_results(output_file_prefix, model_names):
    """加载已存在的结果，返回每个模型已处理的数据ID集合"""
    existing_results = {}
    
    for model_name in model_names:
        output_file = f"{output_file_prefix}_{model_name}.jsonl"
        processed_ids = set()
        
        if os.path.exists(output_file):
            try:
                results = read_jsonl(output_file)
                # 兼容两种格式：artifacts_bench格式使用index，原格式使用custom_id
                processed_ids = {str(item.get('index', item.get('custom_id', ''))) for item in results if item.get('index') or item.get('custom_id')}
                print(f"发现 {model_name} 已处理 {len(processed_ids)} 条数据")
            except Exception as e:
                print(f"读取 {model_name} 现有结果时出错: {e}")
        
        existing_results[model_name] = processed_ids
    
    return existing_results

class ChatBot(object):
    """
    聊天机器人类，支持多种API类型
    api参数格式：
    {
        "type": "OPENAI" | "DASHSCOPE" | "zhipu",
        "base": "API base URL",
        "key": "API key",
        "engine": "模型名称",
        "max_tokens": 输出长度限制（可选）
    }
    """
    def __init__(self, api, max_try=10, max_tokens=4096, tem=0.000001) -> None:
        self.mode = api["type"]
        self.model = api["engine"]
        # self.max_tokens = max_tokens
        self.max_tokens = 8096
        self.max_try = max_try
        self.tem = tem
        self.key = api["key"]
        
        # 如果API中指定了max_tokens则使用API中的值
        if "max_tokens" in api.keys():
            self.max_tokens = api["max_tokens"]
        
        # 初始化不同类型的客户端
        if self.mode == "OPENAI":
            self.client = OpenAI(api_key=api["key"], base_url=api["base"])
    
    def call(self, messages, test=False):
        """调用聊天API，返回(response_content, input_tokens, output_tokens)"""
        if self.mode in ["OPENAI"]:
            return self.call_openai(messages, test)
    
    def call_openai(self, messages, test=False):
        """调用OpenAI API，返回(response_content, input_tokens, output_tokens)"""
        for attempt in range(self.max_try):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.tem,
                    max_tokens=self.max_tokens
                )
                
                # 获取响应内容和token使用信息
                response_content = response.choices[0].message.content
                input_tokens = response.usage.prompt_tokens if response.usage else 0
                output_tokens = response.usage.completion_tokens if response.usage else 0
                
                return response_content, input_tokens, output_tokens
                
            except (RateLimitError, InternalServerError, APITimeoutError) as e:
                if attempt < self.max_try - 1:
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"API错误，第{attempt+1}次重试，等待{wait_time:.2f}秒: {e}")
                    time.sleep(wait_time)
                else:
                    print(f"API调用失败，已重试{self.max_try}次: {e}")
                    return None, 0, 0
            except Exception as e:
                print(f"OpenAI API调用失败: {e}")
                return None, 0, 0

def call_single_model_artifacts(model_name, chat_bot, messages, custom_id, original_data, max_retry=3):
    """适用于artifacts_bench数据格式的单个模型调用函数"""
    thread_id = threading.current_thread().ident
    print(f"[线程{thread_id}] 开始调用 {model_name} 模型处理数据 {custom_id}")
    
    for attempt in range(max_retry):
        try:
            result = chat_bot.call(messages)
            
            if result[0] and result[0].strip():  # result[0] 是响应内容
                response_content, input_tokens, output_tokens = result
                
                print(f"[线程{thread_id}] {model_name} 模型处理数据 {custom_id} 成功")
                print(f"[线程{thread_id}] 输入tokens: {input_tokens}, 输出tokens: {output_tokens}")
                
                return {
                    'index': custom_id,
                    'question': original_data['question'],
                    'model_name': model_name,
                    'response': response_content,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'success': True,
                    'processed_at': datetime.now().isoformat(),
                    'class': original_data.get('class', ''),
                    'difficulty': original_data.get('difficulty', '')
                }
            else:
                print(f"[线程{thread_id}] {model_name} 第 {attempt + 1} 次尝试返回空响应")
                
        except Exception as e:
            print(f"[线程{thread_id}] {model_name} 第 {attempt + 1} 次尝试失败: {str(e)}")
            if attempt < max_retry - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"[线程{thread_id}] 等待 {wait_time:.2f} 秒后重试...")
                time.sleep(wait_time)
    
    # 所有尝试都失败了
    print(f"[线程{thread_id}] {model_name} 处理数据 {custom_id} 经过 {max_retry} 次尝试后仍然失败")
    return {
        'index': custom_id,
        'question': original_data['question'],
        'model_name': model_name,
        'response': None,
        'input_tokens': 0,
        'output_tokens': 0,
        'success': False,
        'processed_at': datetime.now().isoformat(),
        'class': original_data.get('class', ''),
        'difficulty': original_data.get('difficulty', '')
    }

def process_artifacts_bench_file(input_file, output_file_prefix, chat_bots, max_concurrent_requests=6):
    """
    处理artifacts_bench.jsonl文件，使用多线程并行处理多条数据
    优化单模型的处理速度
    """
    # 读取输入数据
    print(f"开始读取文件: {input_file}")
    input_data = read_jsonl(input_file)  # 读取JSONL文件
    print(f"共读取到 {len(input_data)} 条数据")
    
    # 加载已存在的结果
    existing_results = load_existing_results(output_file_prefix, chat_bots.keys())
    
    # 为每个模型创建结果列表，并加载已有结果
    model_results = {}
    for model_name in chat_bots.keys():
        output_file = f"{output_file_prefix}_{model_name}.jsonl"
        if os.path.exists(output_file):
            try:
                model_results[model_name] = read_jsonl(output_file)
                print(f"加载 {model_name} 已有 {len(model_results[model_name])} 条结果")
            except:
                model_results[model_name] = []
        else:
            model_results[model_name] = []
    
    print(f"开始处理文件: {input_file}")
    print(f"将为每个模型创建单独的输出文件:")
    for model_name in chat_bots.keys():
        output_file = f"{output_file_prefix}_{model_name}.jsonl"
        print(f"  {model_name}: {output_file}")
    
    print(f"使用 {max_concurrent_requests} 个并发线程处理数据")
    
    # 使用线程锁保护写入操作
    write_lock = threading.Lock()
    
    # 准备需要处理的任务
    tasks_to_process = []
    skipped_count = 0
    # input_data = [input_data[1458],]
    print(len(input_data),'-----------------------------------------')
    for idx, data in enumerate(input_data):
        try:
            if 'index' not in data or 'question' not in data:
                print(data,'-----------------------------------------')
                print(f"第 {idx + 1} 条数据格式不正确，跳过")
                continue
            
            custom_id = str(data['index'])  # 使用index作为custom_id
            question = data['question']
            
            # 将question转换为messages格式
            messages = [{"role": "user", "content": question}]
            
            # 检查是否需要处理
            need_process = False
            for model_name in chat_bots.keys():
                if custom_id not in existing_results[model_name]:
                    need_process = True
                    break
            
            if need_process:
                tasks_to_process.append((custom_id, messages, idx + 1, data))
            else:
                print(f"跳过数据 {custom_id}（已处理过）")
                skipped_count += 1
                
        except Exception as e:
            print(f"准备第 {idx + 1} 条数据时发生错误: {e}")
            continue
    
    print(f"准备处理 {len(tasks_to_process)} 条数据，跳过 {skipped_count} 条")
    
    # 并行处理所有任务
    success_count = 0
    failed_count = 0
    
    with ThreadPoolExecutor(max_workers=max_concurrent_requests) as executor:
        # 提交所有任务
        future_to_task = {}
        for custom_id, messages, idx, original_data in tasks_to_process:
            for model_name, chat_bot in chat_bots.items():
                if custom_id not in existing_results[model_name]:
                    future = executor.submit(call_single_model_artifacts, model_name, chat_bot, messages, custom_id, original_data)
                    future_to_task[future] = (model_name, custom_id, idx)
        
        print(f"已提交 {len(future_to_task)} 个任务到线程池")
        
        # 收集结果
        completed_tasks = 0
        for future in as_completed(future_to_task):
            model_name, custom_id, idx = future_to_task[future]
            completed_tasks += 1
            
            try:
                result = future.result()
                
                # 使用锁保护写入操作
                with write_lock:
                    model_results[model_name].append(result)
                    existing_results[model_name].add(custom_id)
                    
                    # 立即保存单个模型的结果
                    output_file = f"{output_file_prefix}_{model_name}.jsonl"
                    save_jsonl(model_results[model_name], output_file)
                    
                    if result['success']:
                        success_count += 1
                        print(f"✅ [{completed_tasks}/{len(future_to_task)}] {model_name} 处理数据 {custom_id} 成功")
                    else:
                        failed_count += 1
                        print(f"❌ [{completed_tasks}/{len(future_to_task)}] {model_name} 处理数据 {custom_id} 失败")
                        
            except Exception as e:
                failed_count += 1
                print(f"❌ [{completed_tasks}/{len(future_to_task)}] 处理 {model_name} 数据 {custom_id} 时发生异常: {e}")
                
                # 创建失败记录
                failure_result = {
                    'index': custom_id,
                    'question': '',
                    'model_name': model_name,
                    'response': None,
                    'input_tokens': 0,
                    'output_tokens': 0,
                    'success': False,
                    'processed_at': datetime.now().isoformat()
                }
                with write_lock:
                    model_results[model_name].append(failure_result)
                    existing_results[model_name].add(custom_id)
                    
                    output_file = f"{output_file_prefix}_{model_name}.jsonl"
                    save_jsonl(model_results[model_name], output_file)
    
    # 确保最终结果已保存（实际上每个回复都已实时保存）
    print(f"\n✅ 确保最终结果已保存...")
    for model_name, results in model_results.items():
        output_file = f"{output_file_prefix}_{model_name}.jsonl"
        save_jsonl(results, output_file)
        print(f"{model_name} 最终结果确认保存到: {output_file} (共 {len(results)} 条)")
    
    # 打印总结
    print(f"\n🎉 并行处理完成！")
    print(f"总共处理: {len(tasks_to_process)} 条数据")
    print(f"成功: {success_count} 条")
    print(f"失败: {failed_count} 条")
    print(f"跳过（已处理）: {skipped_count} 条")
    
    # 为每个模型打印详细统计
    print(f"\n📊 各模型处理统计:")
    for model_name, results in model_results.items():
        successful_results = [r for r in results if r['success']]
        failed_results = [r for r in results if not r['success']]
        total_input_tokens = sum(r.get('input_tokens', 0) for r in results)
        total_output_tokens = sum(r.get('output_tokens', 0) for r in results)
        print(f"  {model_name}: 成功 {len(successful_results)} 条, 失败 {len(failed_results)} 条")
        print(f"    输入tokens: {total_input_tokens}, 输出tokens: {total_output_tokens}")

# 主程序
if __name__ == '__main__':
    # 配置模型APIs - 单模型并行优化
    model_configs = {
        # "deepseek-r1-250528": {
        #     "type": "OPENAI", 
        #     "base": 'https://api-gateway.glm.ai/v1',
        #     "key": "sk-IymdOoiHI3umkLDLVToHmBwyfIKoiWoA",
        #     "engine": "deepseek-r1-250528",
        # },
        # "deepseek-v3-0324": {
        #     "type": "OPENAI", 
        #     "base": 'https://api-gateway.glm.ai/v1',
        #     "key": "sk-IymdOoiHI3umkLDLVToHmBwyfIKoiWoA",
        #     "engine": "deepseek-v3-0324",
        # },
        "qwen3-8b-html-sft_full": {
            "type": "OPENAI", 
            "base": "http://localhost:8000/v1",
            "key": "EMPTY",
            "engine": "/workspace/zhengda/self_learn/output/qwen3-8b-html-sft_full",
        },

    }

    # 创建ChatBot实例
    chat_bots = {}
    for model_name, config in model_configs.items():
        chat_bots[model_name] = ChatBot(config)
    
    # 测试连接
    print("测试模型连接...")
    test_messages = [{"role": "user", "content": "你好，请简单回复一下。"}]
    
    for model_name, chat_bot in chat_bots.items():
        try:
            test_result = chat_bot.call(test_messages)
            if test_result[0]:  # test_result[0] 是响应内容
                print(f"{model_name} 连接测试成功: {test_result[0]}")
                print(f"  测试消耗tokens - 输入: {test_result[1]}, 输出: {test_result[2]}")
            else:
                print(f"{model_name} 连接测试失败")
        except Exception as e:
            print(f"{model_name} 连接测试失败: {e}")
    
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 设置文件路径 - 读取artifacts_bench.jsonl文件
    input_file = os.path.join(script_dir, "artifacts_bench.jsonl")  # 输入文件路径
    output_file_prefix = os.path.join(script_dir, "artifacts_results")  # 输出文件前缀
    
    print(f"脚本目录: {script_dir}")
    print(f"输入文件: {input_file}")
    print(f"输出文件前缀: {output_file_prefix}")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 输入文件 '{input_file}' 不存在")
        sys.exit(1)
    
    # 开始处理 - 使用10个并发线程加速
    process_artifacts_bench_file(input_file, output_file_prefix, chat_bots, max_concurrent_requests=10) 