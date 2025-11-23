#!/usr/bin/env python3
"""
組件化演化計算框架 - 統一入口點

這個入口點實現了方案 C：組件化架構，提供了一個統一的接口來運行演化實驗。
支持通過 JSON 配置文件來配置所有演化參數和組件。

使用方式:
    python main_evolution.py --config configs/test_config.json --test
    python main_evolution.py --config configs/portfolio_config.json
"""

import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# 添加項目根目錄到 Python 路徑
sys.path.insert(0, str(Path(__file__).parent))

def load_config(config_path: str) -> Dict[str, Any]:
    """
    載入配置文件
    
    Args:
        config_path: 配置文件路徑
        
    Returns:
        配置字典
    """
    print(f"📄 載入配置文件: {config_path}")
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    print(f"✅ 配置載入成功: {config['experiment']['name']}")
    return config

def load_portfolio_data(data_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    載入投資組合數據
    
    Args:
        data_config: 數據配置
        
    Returns:
        載入的數據字典，包含 train_data 和 test_data
    """
    print(f"📊 載入投資組合數據...")
    
    # 使用現有的數據載入邏輯
    from gp_quant.data.loader import load_and_process_data, split_train_test_data
    import os
    
    # 從 TSE300_selected 目錄載入數據
    tickers_dir = Path(data_config['tickers_dir'])
    if not tickers_dir.exists():
        raise FileNotFoundError(f"數據目錄不存在: {tickers_dir}")
    
    # 獲取所有可用的股票代碼
    csv_files = [f for f in os.listdir(tickers_dir) if f.endswith('.csv')]
    tickers = [f.replace('.csv', '') for f in csv_files]
    
    if not tickers:
        raise ValueError(f"在 {tickers_dir} 中未找到任何 CSV 文件")
    
    print(f"   發現 {len(tickers)} 個股票: {tickers[:3]}{'...' if len(tickers) > 3 else ''}")
    
    # 載入原始數據
    raw_data = load_and_process_data(str(tickers_dir), tickers)
    
    # 分割訓練和測試數據
    train_data, test_data = split_train_test_data(
        raw_data,
        train_data_start=data_config['train_data_start'],
        train_backtest_start=data_config['train_backtest_start'],
        train_backtest_end=data_config['train_backtest_end'],
        test_data_start=data_config['test_data_start'],
        test_backtest_start=data_config['test_backtest_start'],
        test_backtest_end=data_config['test_backtest_end']
    )
    
    data = {
        'train_data': train_data,
        'test_data': test_data,
        'tickers': tickers
    }
    
    print(f"✅ 數據載入完成: {len(tickers)} 個股票")
    return data

def print_experiment_info(config: Dict[str, Any]):
    """打印實驗信息"""
    print("\n" + "🚀" * 60)
    print(f"🧬 組件化演化計算實驗")
    print("🚀" * 60)
    print(f"📋 實驗名稱: {config['experiment']['name']}")
    print(f"📝 實驗描述: {config['experiment']['description']}")
    print(f"📊 數據模式: {config['data']['mode']}")
    print(f"🔢 族群大小: {config['evolution']['population_size']}")
    print(f"🔄 演化世代: {config['evolution']['generations']}")
    print(f"🎯 適應度函數: {config['fitness']['function']}")
    print(f"⚡ 最大處理器: {config['evolution']['max_processors']}")
    print(f"📁 記錄目錄: {config['logging']['records_dir']}")
    
    if config['termination']['early_stopping']:
        print(f"🛑 早停機制: 啟用 (patience={config['termination']['parameters']['patience']})")
    else:
        print(f"🛑 早停機制: 停用")
    
    print("🚀" * 60 + "\n")

def setup_deap_creator():
    """初始化 DEAP creator"""
    from deap import creator, base
    
    # 檢查是否已經創建
    if not hasattr(creator, 'FitnessMax'):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    
    if not hasattr(creator, 'Individual'):
        from deap import gp
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

def main():
    """主函數 - 演化計算入口點"""
    
    # 設置 DEAP creator
    setup_deap_creator()
    
    # 解析命令行參數
    parser = argparse.ArgumentParser(
        description='組件化演化計算框架',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python main_evolution.py --config configs/test_config.json --test
  python main_evolution.py --config configs/portfolio_config.json
        """
    )
    parser.add_argument('--config', required=True, help='配置文件路徑')
    parser.add_argument('--test', action='store_true', help='測試模式 (覆蓋為小規模參數)')
    parser.add_argument('--verbose', '-v', action='store_true', help='詳細輸出模式')
    
    args = parser.parse_args()
    
    try:
        # 1. 載入配置
        config = load_config(args.config)
        
        # 2. 測試模式：覆蓋參數
        if args.test:
            print("🧪 測試模式啟用")
            config['evolution']['population_size'] = 100
            config['evolution']['generations'] = 10
            config['logging']['records_dir'] = 'test_evolution_records'
            config['termination']['parameters']['patience'] = 5
            print(f"   ├─ 族群大小: {config['evolution']['population_size']}")
            print(f"   ├─ 演化世代: {config['evolution']['generations']}")
            print(f"   └─ 記錄目錄: {config['logging']['records_dir']}")
        
        # 3. 打印實驗信息
        print_experiment_info(config)
        
        # 4. 創建記錄目錄
        records_dir = Path(config['logging']['records_dir'])
        records_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 記錄目錄已創建: {records_dir}")
        
        # 5. 載入數據
        data = load_portfolio_data(config['data'])
        
        # 6. 創建演化引擎
        print(f"🏗️ 創建組件化演化引擎...")
        from gp_quant.evolution.components import create_evolution_engine
        
        engine = create_evolution_engine(config)
        print(f"✅ 演化引擎創建完成")
        
        # 7. 執行演化
        print(f"\n🚀 開始演化計算...")
        start_time = datetime.now()
        
        result = engine.evolve(data)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 設置執行時間
        result.execution_time = duration
        
        # 8. 輸出結果
        print(f"\n✅ 演化計算完成!")
        print(f"⏱️  總執行時間: {duration:.2f} 秒 ({duration/60:.2f} 分鐘)")
        print(f"📈 最終世代: {result.final_generation}")
        print(f"🏆 最佳適應度: {result.best_fitness:.4f}")
        print(f"📁 記錄保存於: {config['logging']['records_dir']}")
        if result.genealogy:
            print(f"🧬 個體譜系記錄: {len(result.genealogy)} 個個體")
        else:
            print(f"🧬 個體譜系記錄: 未啟用")
        
        # 9. 保存最終結果摘要
        summary = {
            'experiment_name': config['experiment']['name'],
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'duration_seconds': duration,
            'final_generation': result.final_generation,
            'best_fitness': result.best_fitness,
            'population_size': config['evolution']['population_size'],
            'total_individuals_created': len(result.genealogy) if result.genealogy else 0,
            'config': config
        }
        
        summary_file = Path(config['logging']['records_dir']) / 'experiment_summary.json'
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"📄 實驗摘要保存於: {summary_file}")
        
        return result
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 用戶中斷實驗")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 實驗執行失敗: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
