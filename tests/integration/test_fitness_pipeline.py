#!/usr/bin/env python3
"""
測試適應度評估流程
逐步驗證：信號生成 -> PnL 計算 -> Excess Return 計算
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from deap import creator, base, gp

from gp_quant.data.loader import load_and_process_data, split_train_test_data
from gp_quant.evolution.components.gp import pset
from gp_quant.backtesting.portfolio_engine import PortfolioBacktestingEngine


def setup_deap_creator():
    """初始化 DEAP creator"""
    if not hasattr(creator, 'FitnessMax'):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    
    if not hasattr(creator, 'Individual'):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


def create_test_individuals(n=5):
    """創建測試個體 - 嘗試找到有變化信號的個體"""
    from gp_quant.evolution.components.individual import create_individual
    from gp_quant.evolution.components.gp.operators import NumVector
    import numpy as np
    
    individuals = []
    attempts = 0
    max_attempts = 100
    
    print(f"🔍 嘗試生成 {n} 個有變化信號的個體...")
    
    while len(individuals) < n and attempts < max_attempts:
        attempts += 1
        # 使用更大的深度範圍
        expr = gp.genHalfAndHalf(pset, min_=3, max_=6)
        individual = create_individual(expr)
        
        # 快速檢查是否有變化的信號（使用隨機數據）
        try:
            func = gp.compile(expr=individual, pset=pset)
            test_vec = np.random.randn(100).view(NumVector)
            signal = func(test_vec, test_vec)
            
            if isinstance(signal, np.ndarray):
                unique = np.unique(signal)
                if len(unique) > 1:
                    individuals.append(individual)
                    print(f"   ✅ 找到第 {len(individuals)} 個有效個體（嘗試 {attempts} 次）")
        except:
            pass
    
    if len(individuals) < n:
        print(f"   ⚠️ 只找到 {len(individuals)} 個有效個體（共嘗試 {attempts} 次）")
    else:
        print(f"✅ 成功創建了 {len(individuals)} 個測試個體")
    
    return individuals


def save_signals_to_file(signals, individual_id, output_dir):
    """保存信號到文件"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存每個股票的信號
    for ticker, signal_data in signals.items():
        try:
            if isinstance(signal_data, dict):
                signal_df = pd.DataFrame(signal_data)
            else:
                signal_df = pd.DataFrame({'signals': signal_data})
            
            output_file = output_dir / f"{individual_id[:8]}_{ticker}_signals.csv"
            signal_df.to_csv(output_file)
            print(f"   📁 保存信號: {output_file}")
        except Exception as e:
            print(f"   ⚠️ 無法保存 {ticker} 信號: {e}")


def test_individual_step_by_step(individual, train_data, config, output_dir="test_signals"):
    """逐步測試單個個體的評估流程"""
    
    print("\n" + "=" * 80)
    print(f"🧬 測試個體: {individual.id[:8]}...")
    print("=" * 80)
    
    # 顯示個體表達式
    print(f"\n📝 個體表達式:")
    print(f"   {str(individual)[:200]}...")
    print(f"   樹大小: {len(individual)}, 深度: {individual.height}")
    
    # 處理數據格式
    processed_data = {}
    for ticker, ticker_data in train_data.items():
        if isinstance(ticker_data, dict) and 'data' in ticker_data:
            processed_data[ticker] = ticker_data['data']
        else:
            processed_data[ticker] = ticker_data
    
    # 創建回測引擎
    engine = PortfolioBacktestingEngine(
        data=processed_data,
        backtest_start=config['data']['train_backtest_start'],
        backtest_end=config['data']['train_backtest_end'],
        initial_capital=100000.0,
        pset=pset
    )
    
    print(f"\n✅ 回測引擎創建成功")
    print(f"   股票數量: {len(processed_data)}")
    print(f"   回測期間: {config['data']['train_backtest_start']} 到 {config['data']['train_backtest_end']}")
    
    # ========================================
    # 步驟 1: 生成信號
    # ========================================
    print(f"\n{'='*60}")
    print("📊 步驟 1: 生成交易信號")
    print(f"{'='*60}")
    
    try:
        # 調用內部方法生成信號
        signals = engine._generate_signals_for_all_stocks(individual)
        
        print(f"✅ 信號生成成功!")
        print(f"   股票數量: {len(signals)}")
        
        # 分析每個股票的信號
        for ticker, signal_data in signals.items():
            print(f"\n   📈 {ticker}:")
            print(f"      信號類型: {type(signal_data)}")
            
            if isinstance(signal_data, dict):
                print(f"      信號鍵: {list(signal_data.keys())}")
                
                # 檢查信號內容
                if 'signals' in signal_data:
                    signals_array = signal_data['signals']
                    print(f"      信號數組形狀: {signals_array.shape if hasattr(signals_array, 'shape') else len(signals_array)}")
                    print(f"      信號類型: {type(signals_array)}")
                    print(f"      信號前10個值: {signals_array[:10] if hasattr(signals_array, '__getitem__') else signals_array}")
                    
                    # 統計信號
                    if hasattr(signals_array, '__len__') and len(signals_array) > 0:
                        try:
                            unique_signals = pd.Series(signals_array).value_counts()
                            print(f"      信號統計:")
                            for sig_val, count in unique_signals.items():
                                print(f"         {sig_val}: {count} 次 ({count/len(signals_array)*100:.1f}%)")
                        except:
                            print(f"      無法統計信號（可能是標量值）")
                            print(f"      信號值: {signals_array}")
                
                if 'dates' in signal_data:
                    dates = signal_data['dates']
                    print(f"      日期範圍: {dates[0]} 到 {dates[-1]}")
                    print(f"      交易日數: {len(dates)}")
        
        # 保存信號到文件
        save_signals_to_file(signals, individual.id, output_dir)
        
    except Exception as e:
        print(f"❌ 信號生成失敗: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # ========================================
    # 步驟 2: 執行回測並計算 PnL
    # ========================================
    print(f"\n{'='*60}")
    print("💰 步驟 2: 執行回測並計算 PnL")
    print(f"{'='*60}")
    
    try:
        # 執行完整回測
        backtest_results = engine.backtest(individual)
        
        print(f"✅ 回測執行成功!")
        print(f"   結果鍵: {list(backtest_results.keys())}")
        
        # 檢查 equity curve
        equity_curve = backtest_results.get('equity_curve')
        if equity_curve is not None:
            print(f"\n   📈 投資組合價值曲線:")
            print(f"      長度: {len(equity_curve)}")
            print(f"      初始值: ${equity_curve.iloc[0]:,.2f}")
            print(f"      最終值: ${equity_curve.iloc[-1]:,.2f}")
            print(f"      最大值: ${equity_curve.max():,.2f}")
            print(f"      最小值: ${equity_curve.min():,.2f}")
            
            # 計算變化
            total_change = equity_curve.iloc[-1] - equity_curve.iloc[0]
            total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0] - 1) * 100
            print(f"      總變化: ${total_change:,.2f}")
            print(f"      總報酬率: {total_return:.2f}%")
        
        # 檢查每股 PnL
        per_stock_pnl = backtest_results.get('per_stock_pnl', {})
        print(f"\n   💵 每股 PnL:")
        for ticker, pnl in per_stock_pnl.items():
            print(f"      {ticker}: ${pnl:,.2f}")
        
        total_pnl = sum(per_stock_pnl.values())
        print(f"      總 PnL: ${total_pnl:,.2f}")
        
        # 檢查交易記錄
        transactions = backtest_results.get('transactions')
        if transactions is not None and len(transactions) > 0:
            print(f"\n   💼 交易記錄:")
            print(f"      交易次數: {len(transactions)}")
            print(f"      前5筆交易:")
            print(transactions.head())
        else:
            print(f"\n   ⚠️ 沒有交易記錄")
        
        # 保存 equity curve
        output_dir = Path(output_dir)
        equity_file = output_dir / f"{individual.id[:8]}_equity_curve.csv"
        equity_curve.to_csv(equity_file)
        print(f"\n   📁 保存 equity curve: {equity_file}")
        
    except Exception as e:
        print(f"❌ 回測執行失敗: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # ========================================
    # 步驟 3: 計算 Excess Return
    # ========================================
    print(f"\n{'='*60}")
    print("🎯 步驟 3: 計算 Excess Return (適應度)")
    print(f"{'='*60}")
    
    try:
        metrics = backtest_results.get('metrics', {})
        
        print(f"✅ 績效指標:")
        print(f"   總報酬率: {metrics.get('total_return', 0):.4f}%")
        print(f"   超額報酬: {metrics.get('excess_return', 0):.4f}")
        print(f"   夏普比率: {metrics.get('sharpe_ratio', 0):.4f}")
        print(f"   最大回撤: {metrics.get('max_drawdown', 0):.4f}%")
        print(f"   波動率: {metrics.get('volatility', 0):.4f}%")
        print(f"   卡瑪比率: {metrics.get('calmar_ratio', 0):.4f}")
        print(f"   勝率: {metrics.get('win_rate', 0):.4f}%")
        
        # 使用 get_fitness 方法
        fitness_value = engine.get_fitness(individual, fitness_metric='excess_return')
        print(f"\n🏆 最終適應度 (excess_return): {fitness_value:.6f}")
        
        return {
            'individual_id': individual.id,
            'signals': signals,
            'backtest_results': backtest_results,
            'fitness': fitness_value
        }
        
    except Exception as e:
        print(f"❌ 適應度計算失敗: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """主函數"""
    print("=" * 80)
    print("🧪 適應度評估流程測試")
    print("=" * 80)
    
    # 設置 DEAP
    setup_deap_creator()
    
    # 載入配置
    config_path = "configs/test_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"\n📄 載入配置: {config_path}")
    
    # 載入數據
    print(f"\n📊 載入數據...")
    tickers_dir = Path(config['data']['tickers_dir'])
    if not tickers_dir.is_absolute():
        tickers_dir = Path.cwd() / tickers_dir
    
    tickers = [f.stem for f in tickers_dir.glob("*.csv")]
    raw_data = load_and_process_data(str(tickers_dir), tickers)
    
    train_data, test_data = split_train_test_data(
        raw_data,
        train_data_start=config['data']['train_data_start'],
        train_backtest_start=config['data']['train_backtest_start'],
        train_backtest_end=config['data']['train_backtest_end'],
        test_data_start=config['data']['test_data_start'],
        test_backtest_start=config['data']['test_backtest_start'],
        test_backtest_end=config['data']['test_backtest_end']
    )
    
    print(f"✅ 數據載入完成: {len(train_data)} 個股票")
    
    # 創建測試個體
    print(f"\n🧬 創建測試個體...")
    individuals = create_test_individuals(n=3)
    
    # 測試每個個體
    results = []
    for i, individual in enumerate(individuals):
        result = test_individual_step_by_step(
            individual, 
            train_data, 
            config,
            output_dir=f"test_signals/individual_{i+1}"
        )
        if result:
            results.append(result)
    
    # 總結
    print("\n" + "=" * 80)
    print("📊 測試總結")
    print("=" * 80)
    
    if results:
        print(f"\n✅ 成功測試了 {len(results)} 個個體")
        print(f"\n🏆 適應度排名:")
        sorted_results = sorted(results, key=lambda x: x['fitness'], reverse=True)
        for i, result in enumerate(sorted_results):
            print(f"   {i+1}. ID: {result['individual_id'][:8]}..., Fitness: {result['fitness']:.6f}")
    else:
        print(f"\n❌ 所有個體測試失敗")
    
    print(f"\n📁 信號和結果已保存到 test_signals/ 目錄")


if __name__ == "__main__":
    main()
