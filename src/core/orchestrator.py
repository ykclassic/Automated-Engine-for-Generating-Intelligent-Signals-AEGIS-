"""
AEGIS Main Orchestrator
Coordinates all components for end-to-end operation
"""

import logging
import json
from typing import Dict, List, Optional
from datetime import datetime

# FIXED: Absolute imports for GitHub Actions
try:
    from core.data_fetcher import DataPipeline
    from core.signal_generator import SignalGenerator, TradingSignal
    from core.risk_manager import RiskManager, RiskLevel
    from utils.performance_tracker import PerformanceTracker
    from utils.system_monitor import SystemMonitor
    from utils.optimizer import StrategyOptimizer
    from notifications.formatter import SignalFormatter
except ImportError:
    from src.core.data_fetcher import DataPipeline
    from src.core.signal_generator import SignalGenerator, TradingSignal
    from src.core.risk_manager import RiskManager, RiskLevel
    from src.utils.performance_tracker import PerformanceTracker
    from src.utils.system_monitor import SystemMonitor
    from src.utils.optimizer import StrategyOptimizer
    from src.notifications.formatter import SignalFormatter

logger = logging.getLogger(__name__)

class AEGISOrchestrator:
    def __init__(self, risk_level: RiskLevel = RiskLevel.MODERATE, account_balance: float = 10000):
        self.risk_level = risk_level
        self.account_balance = account_balance
        
        self.data_pipeline = DataPipeline()
        self.signal_generator = SignalGenerator(risk_level=risk_level)
        self.risk_manager = RiskManager(risk_level=risk_level)
        self.performance_tracker = PerformanceTracker()
        self.system_monitor = SystemMonitor()
        self.optimizer = StrategyOptimizer()
        self.formatter = SignalFormatter()
        
        logger.info("🛡️ AEGIS Orchestrator initialized")
    
    def run_cycle(self) -> Dict:
        cycle_start = datetime.now()
        logger.info(f"Starting cycle at {cycle_start}")
        
        results = {
            'timestamp': cycle_start.isoformat(),
            'status': 'running',
            'signals_generated': 0,
            'errors': []
        }
        
        try:
            # 1. Check system health
            health = self.system_monitor.get_health_summary()
            if health.get('overall_status') == 'critical':
                results['status'] = 'aborted'
                return results
            
            # 2. Fetch data
            all_data = self.data_pipeline.fetch_all_assets()
            if not all_data:
                results['errors'].append("No data fetched")
                return results
            
            # 3. Generate signals
            signals = self.signal_generator.generate_all_signals(all_data, self.account_balance)
            results['signals_generated'] = len(signals)
            
            # 4. Process & Record
            for signal in signals:
                self.performance_tracker.record_signal({
                    'symbol': signal.symbol,
                    'direction': signal.direction,
                    'entry_price': signal.entry_price,
                    'timestamp': signal.timestamp.isoformat()
                })
            
            results['status'] = 'success'
            results['duration_seconds'] = (datetime.now() - cycle_start).total_seconds()
            
        except Exception as e:
            logger.error(f"Cycle error: {e}")
            results['status'] = 'error'
            results['errors'].append(str(e))
        
        return results

def main():
    orchestrator = AEGISOrchestrator()
    results = orchestrator.run_cycle()
    print(json.dumps(results, indent=2, default=str))

if __name__ == "__main__":
    main()
