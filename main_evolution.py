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
import copy

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


def load_explicit_data(train_csv: str, test_csv: str) -> Dict[str, Any]:
    """
    從明確指定的 CSV 檔案載入訓練和測試資料。
    
    用於多資產多 fold 實驗，直接使用 regime_splits 的 CSV 檔案。
    
    Args:
        train_csv: 訓練資料 CSV 檔案路徑
        test_csv: 測試資料 CSV 檔案路徑
        
    Returns:
        資料字典，包含 train_data, test_data, tickers, date_metadata
    """
    import pandas as pd
    
    print(f"📊 載入明確指定的資料檔案...")
    print(f"   Train: {train_csv}")
    print(f"   Test: {test_csv}")
    
    train_path = Path(train_csv)
    test_path = Path(test_csv)
    
    if not train_path.exists():
        raise FileNotFoundError(f"訓練資料檔案不存在: {train_csv}")
    if not test_path.exists():
        raise FileNotFoundError(f"測試資料檔案不存在: {test_csv}")
    
    # 從檔名推斷 ticker 名稱
    ticker = train_path.parent.name.upper()  # e.g., "btc_usd" -> "BTC_USD"
    
    # 載入訓練資料
    train_df = pd.read_csv(train_path, parse_dates=['Date'], index_col='Date')
    if hasattr(train_df.index, 'tz') and train_df.index.tz is not None:
        train_df.index = train_df.index.tz_convert(None)
    train_df.sort_index(inplace=True)
    
    # 載入測試資料
    test_df = pd.read_csv(test_path, parse_dates=['Date'], index_col='Date')
    if hasattr(test_df.index, 'tz') and test_df.index.tz is not None:
        test_df.index = test_df.index.tz_convert(None)
    test_df.sort_index(inplace=True)
    
    # 建立資料結構（與 split_train_test_data 相同格式）
    train_start = train_df.index[0]
    train_end = train_df.index[-1]
    test_start = test_df.index[0]
    test_end = test_df.index[-1]
    
    # 計算 warmup 期間（前 250 天用於技術指標計算）
    warmup_days = min(250, len(train_df) // 4)
    train_backtest_start = train_df.index[warmup_days]
    test_backtest_start = test_df.index[min(warmup_days, len(test_df) // 4)]
    
    train_data = {
        ticker: {
            'data': train_df,
            'backtest_start': str(train_backtest_start.date()),
            'backtest_end': str(train_end.date()),
        }
    }
    
    test_data = {
        ticker: {
            'data': test_df,
            'backtest_start': str(test_backtest_start.date()),
            'backtest_end': str(test_end.date()),
        }
    }
    
    # 日期元資料 (用於更新 config)
    date_metadata = {
        'train_data_start': str(train_start.date()),
        'train_backtest_start': str(train_backtest_start.date()),
        'train_backtest_end': str(train_end.date()),
        'test_data_start': str(test_start.date()),
        'test_backtest_start': str(test_backtest_start.date()),
        'test_backtest_end': str(test_end.date()),
    }
    
    print(f"   Ticker: {ticker}")
    print(f"   Train: {len(train_df)} days ({train_start.date()} ~ {train_end.date()})")
    print(f"   Train backtest: {train_backtest_start.date()} ~ {train_end.date()}")
    print(f"   Test: {len(test_df)} days ({test_start.date()} ~ {test_end.date()})")
    print(f"✅ 資料載入完成")
    
    return {
        'train_data': train_data,
        'test_data': test_data,
        'tickers': [ticker],
        'date_metadata': date_metadata,
    }

def load_portfolio_data(data_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    載入投資組合數據
    
    Args:
        data_config: 數據配置
        
    Returns:
        載入的數據字典，包含 train_data, test_data, 和可選的 validate_data
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
    
    # 檢查是否有 validate 配置（向下兼容）
    has_validate = all([
        data_config.get('validate_data_start'),
        data_config.get('validate_backtest_start'),
        data_config.get('validate_backtest_end')
    ])
    
    # 分割訓練、驗證和測試數據
    train_data, test_data, validate_data = split_train_test_data(
        raw_data,
        train_data_start=data_config['train_data_start'],
        train_backtest_start=data_config['train_backtest_start'],
        train_backtest_end=data_config['train_backtest_end'],
        test_data_start=data_config['test_data_start'],
        test_backtest_start=data_config['test_backtest_start'],
        test_backtest_end=data_config['test_backtest_end'],
        # Optional validate parameters
        validate_data_start=data_config.get('validate_data_start'),
        validate_backtest_start=data_config.get('validate_backtest_start'),
        validate_backtest_end=data_config.get('validate_backtest_end')
    )
    
    data = {
        'train_data': train_data,
        'test_data': test_data,
        'tickers': tickers
    }
    
    # 只有在有 validate 配置時才加入
    if validate_data:
        data['validate_data'] = validate_data
        print(f"✅ 數據載入完成: {len(tickers)} 個股票 (Train + Validate + Test)")
    else:
        print(f"✅ 數據載入完成: {len(tickers)} 個股票 (Train + Test)")
    
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
    parser.add_argument('--no-timestamp', action='store_true', help='不添加時間流水號到記錄目錄')
    
    # New arguments for explicit data paths
    parser.add_argument('--train-data', type=str, help='明確指定訓練資料 CSV 檔案路徑')
    parser.add_argument('--test-data', type=str, help='明確指定測試資料 CSV 檔案路徑')
    parser.add_argument('--output-dir', type=str, help='覆蓋輸出目錄 (logging.records_dir)')
    
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
        
        # 2.1 覆蓋輸出目錄 (--output-dir)
        if args.output_dir:
            config['logging']['records_dir'] = args.output_dir
            print(f"📁 使用指定輸出目錄: {args.output_dir}")
        
        # 2.5. 添加時間流水號到記錄目錄
        if not args.no_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            original_dir = config['logging']['records_dir']
            config['logging']['records_dir'] = f"{original_dir}_{timestamp}"
            print(f"🕐 添加時間流水號: {original_dir} -> {config['logging']['records_dir']}")
        
        # 3. 打印實驗信息
        print_experiment_info(config)
        
        # 4. 創建記錄目錄
        records_dir = Path(config['logging']['records_dir'])
        records_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 記錄目錄已創建: {records_dir}")
        
        # 5. 載入數據
        if args.train_data and args.test_data:
            # 使用明確指定的 CSV 檔案
            data = load_explicit_data(args.train_data, args.test_data)
            
            # 更新 config 的日期設定 (用於 Rolling Window Evaluator)
            date_meta = data['date_metadata']
            config['data']['train_data_start'] = date_meta['train_data_start']
            config['data']['train_backtest_start'] = date_meta['train_backtest_start']
            config['data']['train_backtest_end'] = date_meta['train_backtest_end']
            config['data']['test_data_start'] = date_meta['test_data_start']
            config['data']['test_backtest_start'] = date_meta['test_backtest_start']
            config['data']['test_backtest_end'] = date_meta['test_backtest_end']
            print(f"📅 已更新 config 日期: train {date_meta['train_backtest_start']} ~ {date_meta['train_backtest_end']}")
            
            # 為 parallel worker 建立 symlink (讓 tickers_dir 能找到資料)
            import os
            import tempfile
            ticker = data['tickers'][0]
            train_path = Path(args.train_data).resolve()
            
            # 建立臨時目錄並 symlink 訓練資料
            temp_tickers_dir = Path(tempfile.mkdtemp(prefix="gp_quant_tickers_"))
            symlink_path = temp_tickers_dir / f"{ticker}.csv"
            symlink_path.symlink_to(train_path)
            
            # 更新 config 的 tickers_dir
            config['data']['tickers_dir'] = str(temp_tickers_dir)
            print(f"🔗 建立 symlink: {symlink_path} -> {train_path}")
            print(f"   Parallel mode enabled with tickers_dir: {temp_tickers_dir}")
        elif args.train_data or args.test_data:
            raise ValueError("必須同時指定 --train-data 和 --test-data")
        else:
            # 使用傳統的資料夾載入方式
            data = load_portfolio_data(config['data'])
        
        # 6. 選擇並創建引擎
        experiment_type = config['experiment'].get('type', 'standard')
        
        if experiment_type == 'walk_forward':
            print(f"🏗️ 創建 Walk-Forward 演化引擎...")
            from gp_quant.backtesting.walk_forward import WalkForwardEvolutionEngine
            engine = WalkForwardEvolutionEngine(config)
            print(f"✅ Walk-Forward 引擎創建完成")
            
            print(f"\n🚀 開始 Walk-Forward 分析...")
            start_time = datetime.now()
            
            # WF engine run returns a dict
            wf_result = engine.run(data)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            if wf_result:
                print(f"\n✅ Walk-Forward 分析完成!")
                print(f"⏱️  總執行時間: {duration:.2f} 秒 ({duration/60:.2f} 分鐘)")
                print(f"📈 處理視窗數: {len(wf_result['window_results'])}")
                print(f"💰 總回報: {wf_result['metrics']['total_return']:.2%}")
                print(f"📊 Sharpe Ratio: {wf_result['metrics']['sharpe_ratio']:.4f}")
                print(f"📉 Max Drawdown: {wf_result['metrics']['max_drawdown']:.2%}")
                
                # 保存結果
                result_file = Path(config['logging']['records_dir']) / 'final_result.json'
                
                # Convert Series to list/dict for JSON serialization
                # We need to be careful with serialization
                serializable_result = copy.deepcopy(wf_result)
                # Convert equity curve to list of [date, value] or just values
                # Actually, let's just save metrics and window summary for now
                # The equity curve is a Series with DatetimeIndex
                
                # Simple serialization helper
                def convert_for_json(obj):
                    if isinstance(obj, pd.Series):
                        return obj.to_dict() # Index (Timestamp) to value
                    if isinstance(obj, pd.Timestamp):
                        return obj.isoformat()
                    return str(obj)

                # Save full result with custom encoder logic or just simplified
                # Let's save a simplified version
                final_output = {
                    'metrics': wf_result['metrics'],
                    'window_results': []
                }
                
                for wr in wf_result['window_results']:
                    win_res = {
                        'window_index': wr['window_index'],
                        'train_period': f"{wr['window']['train_start'].date()} to {wr['window']['train_end'].date()}",
                        'test_period': f"{wr['window']['test_start'].date()} to {wr['window']['test_end'].date()}",
                        'best_fitness': wr['best_fitness'],
                        'oos_metrics': wr['oos_metrics']
                    }
                    final_output['window_results'].append(win_res)

                with open(result_file, 'w', encoding='utf-8') as f:
                    json.dump(final_output, f, indent=2, ensure_ascii=False)
                
                print(f"🏆 最終結果保存於: {result_file}")
                
                # Also save summary
                summary = {
                    'experiment_name': config['experiment']['name'],
                    'type': 'walk_forward',
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'duration_seconds': duration,
                    'metrics': wf_result['metrics'],
                    'config': config
                }
                summary_file = Path(config['logging']['records_dir']) / 'experiment_summary.json'
                with open(summary_file, 'w', encoding='utf-8') as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False)
                print(f"📄 實驗摘要保存於: {summary_file}")
                
                return wf_result
            else:
                print("❌ Walk-Forward 分析未返回結果")
                return None

        else:
            # Standard Evolution
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
