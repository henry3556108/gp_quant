#!/usr/bin/env python3
"""
繼續演化腳本 - 從保存的狀態繼續演化計算
"""
import argparse
import json
from pathlib import Path
from datetime import datetime

from gp_quant.data.loader import load_and_process_data
from gp_quant.evolution.components.loader import EvolutionLoader, analyze_saved_evolution


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
    """主函數"""
    parser = argparse.ArgumentParser(
        description='從保存的狀態繼續演化計算',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用範例:
  python continue_evolution.py --records test_evolution_records --generations 10
  python continue_evolution.py --records test_evolution_records --analyze
        """
    )
    
    parser.add_argument('--records', required=True,
                       help='演化記錄目錄路徑')
    parser.add_argument('--generations', type=int, default=10,
                       help='額外的演化世代數 (默認: 10)')
    parser.add_argument('--analyze', action='store_true',
                       help='只分析保存的演化數據，不繼續演化')
    parser.add_argument('--config', 
                       help='配置文件路徑 (用於載入數據配置)')
    
    args = parser.parse_args()
    
    # 設置 DEAP creator
    setup_deap_creator()
    
    print("🔄" * 60)
    print("🔄 演化狀態重載與繼續")
    print("🔄" * 60)
    
    try:
        # 創建載入器
        loader = EvolutionLoader(args.records)
        
        if args.analyze:
            # 只分析數據
            print("📊 分析保存的演化數據...")
            analysis = analyze_saved_evolution(args.records)
            
            print("\n📈 演化進度分析:")
            print(f"   📊 總世代數: {analysis.get('total_generations', 0)}")
            print(f"   🔢 世代範圍: {analysis.get('generation_range', 'N/A')}")
            print(f"   📋 統計數據: {'✅' if analysis.get('has_statistics', False) else '❌'}")
            
            if 'best_fitness_overall' in analysis:
                print(f"   🏆 最佳適應度: {analysis['best_fitness_overall']:.6f}")
                print(f"   📉 最差適應度: {analysis['worst_fitness_overall']:.6f}")
                print(f"   📈 適應度改進: {analysis['fitness_improvement']:.6f}")
                print(f"   🎯 收斂檢測: {'✅' if analysis.get('convergence_detected', False) else '❌'}")
            
            print(f"\n📁 可用世代: {analysis.get('available_generations', [])}")
            
        else:
            # 繼續演化
            print(f"📁 記錄目錄: {args.records}")
            print(f"➕ 額外世代: {args.generations}")
            
            # 載入配置和數據
            if args.config:
                config_path = Path(args.config)
            else:
                # 嘗試從記錄目錄載入配置
                config_path = Path(args.records) / "config.json"
            
            if not config_path.exists():
                raise FileNotFoundError(f"配置文件不存在: {config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            print(f"📄 載入配置: {config_path}")
            
            # 載入數據
            print("📊 載入數據...")
            tickers_dir = Path(config['data']['tickers_dir'])
            if not tickers_dir.is_absolute():
                tickers_dir = Path.cwd() / tickers_dir
            
            data = load_and_process_data(
                str(tickers_dir),
                mode=config['data']['mode'],
                train_data_start=config['data']['train_data_start'],
                train_backtest_start=config['data']['train_backtest_start'],
                train_backtest_end=config['data']['train_backtest_end'],
                test_data_start=config['data']['test_data_start'],
                test_backtest_start=config['data']['test_backtest_start'],
                test_backtest_end=config['data']['test_backtest_end']
            )
            
            print(f"✅ 數據載入完成: {len(data)} 個股票")
            
            # 繼續演化
            print(f"\n🚀 開始繼續演化...")
            start_time = datetime.now()
            
            result = loader.continue_evolution(args.generations, data)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # 輸出結果
            print(f"\n✅ 演化繼續完成!")
            print(f"⏱️  執行時間: {duration:.2f} 秒 ({duration/60:.2f} 分鐘)")
            print(f"📈 最終世代: {result.final_generation}")
            print(f"🏆 最佳適應度: {result.best_fitness:.4f}")
            
    except Exception as e:
        print(f"❌ 錯誤: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
