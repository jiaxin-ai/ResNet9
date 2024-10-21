import subprocess

def get_gpu_info():
    try:
        # 运行 nvidia-smi 命令，查询显存使用量和 GPU 利用率
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # 检查命令是否成功执行
        if result.returncode != 0:
            print(f"Error running nvidia-smi: {result.stderr}")
            return None

        # 解析输出
        gpu_info = []
        for line in result.stdout.splitlines():
            memory_used, gpu_utilization = line.split(',')
            gpu_info.append({
                'memory_used': int(memory_used.strip()),          # 显存使用 (MB)
                'gpu_utilization': int(gpu_utilization.strip())   # GPU 利用率 (%)
            })

        return gpu_info

    except Exception as e:
        print(f"Exception while querying GPU info: {e}")
        return None

if __name__ == '__main__':
    # 获取 GPU 信息
    gpu_info = get_gpu_info()
    # print(gpu_info[0]['memory_used'],gpu_info[0]['gpu_utilization'])
    # 打印 GPU 信息
    if gpu_info:
        for idx, info in enumerate(gpu_info):
            print(f"GPU {idx}: Memory Used = {info['memory_used']} MB, GPU Utilization = {info['gpu_utilization']}%")
    else:
        print("Failed to retrieve GPU information.")
