import wandb
import yaml
import os

def load_sweep_config(yaml_path="sweep_config.yaml"):
    """从YAML文件加载sweep配置"""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        sweep_config = yaml.safe_load(f)
    return sweep_config

def run_sweep_from_yaml(yaml_path="sweep_config.yaml", project_name="ae-contrastive", count=50):
    """使用YAML配置运行sweep"""
    
    # 1. 登录wandb（首次需要输入API key）
    wandb.login()
    
    # 2. 加载YAML配置
    sweep_config = load_sweep_config(yaml_path)
    print(f"Loaded sweep config from {yaml_path}")
    
    # 3. 创建sweep
    sweep_id = wandb.sweep(sweep=sweep_config, project=project_name)
    print(f"Created sweep: {sweep_id}")
    
    # 4. 运行agent
    print(f"\nStarting {count} runs...")
    wandb.agent(sweep_id, count=count, project=project_name)
    
    print("\nSweep completed!")
    return sweep_id

if __name__ == "__main__":
    # 可以指定不同的配置文件
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='sweep_config.yaml',
                       help='Path to sweep configuration YAML file')
    parser.add_argument('--project', type=str, default='ae-contrastive',
                       help='WandB project name')
    parser.add_argument('--count', type=int, default=50,
                       help='Number of runs to execute')
    args = parser.parse_args()
    
    run_sweep_from_yaml(yaml_path=args.config, project_name=args.project, count=args.count)