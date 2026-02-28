"""
AEGIS Main Orchestrator
Coordinates all components for end-to-end operation
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime

from .data_fetcher import DataPipeline
from .signal_generator import SignalGenerator, TradingSignal
from .risk_manager import RiskManager, RiskLevel
from ..utils.performance_tracker import PerformanceTracker
from ..utils.system_monitor import SystemMonitor
from ..utils.optimizer import StrategyOptimizer
from ..notifications.formatter import SignalFormatter

logger = logging.getLogger(__name__)


class AEGISOrchestrator:
    """
    Main orchestrator for AEGIS system
    """
    
    def __init__(
        self,
        risk_level: RiskLevel = RiskLevel.MODERATE,
        account_balance: float = 10000
    ):
        self.risk_level = risk_level
        self.account_balance = account_balance
        
        # Initialize components
        self.data_pipeline = DataPipeline()
        self.signal_generator = SignalGenerator(risk_level=risk_level)
        self.risk_manager = RiskManager(risk_level=risk_level)
        self.performance_tracker = PerformanceTracker()
        self.system_monitor = SystemMonitor()
        self.optimizer = StrategyOptimizer()
        self.formatter = SignalFormatter()
        
        logger.info("🛡️ AEGIS Orchestrator initialized")
    
    def run_cycle(self) -> Dict:
        """
        Run complete signal generation cycle
        """
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
            if health['overall_status'] == 'critical':
                results['status'] = 'aborted'
                results['reason'] = 'Critical system health'
                return results
            
            # 2. Fetch data for all assets
            logger.info("Fetching market data...")
            all_data = self.data_pipeline.fetch_all_assets()
            
            if not all_data:
                results['errors'].append("No data fetched")
                return results
            
            # 3. Generate signals
            logger.info("Generating signals...")
            signals = self.signal_generator.generate_all_signals(
                all_data, 
                self.account_balance
            )
            
            results['signals_generated'] = len(signals)
            
            # 4. Record signals
            for signal in signals:
                self.performance_tracker.record_signal({
                    'symbol': signal.symbol,
                    'direction': signal.direction,
                    'entry_price': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'take_profit': signal.take_profit,
                    'size': signal.position_size['size_pct'] * self.account_balance,
                    'timestamp': signal.timestamp.isoformat()
                })
            
            # 5. Check for completed trades
            current_prices = {
                symbol: data['1h']['close'].iloc[-1]
                for symbol, data in all_data.items()
                if '1h' in data
            }
            
            completed = self.performance_tracker.update_signal_outcome(
                '', datetime.now(), 0  # Will check all open signals
            )
            
            results['trades_completed'] = len(completed)
            
            # 6. Update performance metrics
            metrics = self.performance_tracker.calculate_metrics()
            results['performance'] = metrics
            
            # 7. Run optimization if needed
            if len(self.performance_tracker.trades) > 20:
                opt_result = self.optimizer.analyze_performance()
                if opt_result['status'] == 'analyzed':
                    results['optimization_suggested'] = True
                    results['optimization_issues'] = opt_result['issues']
            
            # 8. System health update
            self.system_monitor.check_data_pipeline(datetime.now())
            
            results['status'] = 'success'
            results['duration_seconds'] = (datetime.now() - cycle_start).total_seconds()
            
        except Exception as e:
            logger.error(f"Cycle error: {e}")
            results['status'] = 'error'
            results['errors'].append(str(e))
            self.system_monitor.log_event('orchestrator', 'ERROR', str(e))
        
        return results
    
    def get_dashboard_data(self) -> Dict:
        """
        Get data for dashboard display
        """
        return {
            'signals': self.performance_tracker.signals[-20:],
            'performance': self.performance_tracker.calculate_metrics(),
            'system_health': self.system_monitor.get_health_summary(),
            'open_positions': len(self.signal_generator.risk_manager.open_positions),
            'risk_metrics': self.signal_generator.risk_manager.get_risk_metrics().__dict__
        }
    
    def generate_report(self) -> str:
        """
        Generate full system report
        """
        lines = [
            "🛡️ AEGIS SYSTEM REPORT",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "📊 PERFORMANCE",
            "-" * 20,
            self.performance_tracker.generate_report(),
            "",
            "🔧 SYSTEM STATUS",
            "-" * 20,
            f"Overall: {self.system_monitor.get_health_summary()['overall_status']}",
            f"Components: {self.system_monitor.component_status}",
            "",
            "⚠️ RISK METRICS",
            "-" * 20,
        ]
        
        risk = self.signal_generator.risk_manager.get_risk_metrics()
        lines.extend([
            f"Portfolio Heat: {risk.portfolio_heat:.1%}",
            f"Daily VaR: {risk.daily_var:.2%}",
            f"Current Drawdown: {risk.current_drawdown:.2%}",
            f"Sharpe Ratio: {risk.sharpe_ratio:.2f}",
        ])
        
        return "\n".join(lines)


# Main entry point
def main():
    """Main execution"""
    orchestrator = AEGISOrchestrator()
    results = orchestrator.run_cycle()
    print(json.dumps(results, indent=2, default=str))
    
    # Print report if signals generated
    if results['signals_generated'] > 0:
        print("\n" + orchestrator.generate_report())


if __name__ == "__main__":
    import json
    main()
