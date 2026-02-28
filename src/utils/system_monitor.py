"""
AEGIS System Health Monitor
Monitors system components and triggers alerts
"""

import logging
import psutil
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SystemEvent:
    """System event record"""
    timestamp: datetime
    component: str
    level: str  # INFO, WARNING, ERROR, CRITICAL
    message: str
    metric_value: Optional[float] = None


class SystemMonitor:
    """
    Monitors system health and performance
    """
    
    def __init__(self):
        self.events: List[SystemEvent] = []
        self.component_status: Dict[str, str] = {}
        self.alert_thresholds = {
            'cpu_percent': 80,
            'memory_percent': 85,
            'disk_percent': 90,
            'latency_ms': 1000
        }
    
    def check_system_resources(self) -> Dict:
        """
        Check system resource usage
        """
        metrics = {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_percent': psutil.disk_usage('/').percent,
            'network_io': psutil.net_io_counters()._asdict()
        }
        
        # Check thresholds
        for metric, value in metrics.items():
            if metric in self.alert_thresholds:
                if value > self.alert_thresholds[metric]:
                    self.log_event(
                        'system',
                        'WARNING',
                        f'{metric} at {value}%, exceeds threshold {self.alert_thresholds[metric]}%',
                        value
                    )
        
        return metrics
    
    def check_data_pipeline(self, last_fetch_time: Optional[datetime] = None) -> str:
        """
        Check data pipeline health
        """
        if last_fetch_time is None:
            status = 'unknown'
        else:
            minutes_since = (datetime.now() - last_fetch_time).total_seconds() / 60
            
            if minutes_since < 10:
                status = 'healthy'
            elif minutes_since < 30:
                status = 'warning'
                self.log_event('data_pipeline', 'WARNING', 
                              f'Last fetch {minutes_since:.0f} minutes ago')
            else:
                status = 'critical'
                self.log_event('data_pipeline', 'CRITICAL', 
                              f'Data pipeline stalled, last fetch {minutes_since:.0f} minutes ago')
        
        self.component_status['data_pipeline'] = status
        return status
    
    def check_ml_models(self, last_prediction_time: Optional[datetime] = None) -> str:
        """
        Check ML model health
        """
        if last_prediction_time is None:
            status = 'unknown'
        else:
            minutes_since = (datetime.now() - last_prediction_time).total_seconds() / 60
            
            if minutes_since < 5:
                status = 'healthy'
            elif minutes_since < 15:
                status = 'warning'
            else:
                status = 'critical'
                self.log_event('ml_models', 'WARNING',
                              f'ML prediction latency high: {minutes_since:.0f} minutes')
        
        self.component_status['ml_models'] = status
        return status
    
    def log_event(self, component: str, level: str, message: str, metric: Optional[float] = None):
        """
        Log system event
        """
        event = SystemEvent(
            timestamp=datetime.now(),
            component=component,
            level=level,
            message=message,
            metric_value=metric
        )
        
        self.events.append(event)
        
        # Log to system logger
        log_func = getattr(logger, level.lower(), logger.info)
        log_func(f"[{component}] {message}")
        
        # Keep only recent events
        cutoff = datetime.now() - timedelta(days=7)
        self.events = [e for e in self.events if e.timestamp > cutoff]
    
    def get_health_summary(self) -> Dict:
        """
        Get overall health summary
        """
        # Check all components
        resources = self.check_system_resources()
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'overall_status': 'healthy',
            'components': self.component_status,
            'resources': resources,
            'recent_events': [
                {
                    'time': e.timestamp.strftime('%H:%M:%S'),
                    'component': e.component,
                    'level': e.level,
                    'message': e.message
                }
                for e in self.events[-10:]
            ]
        }
        
        # Determine overall status
        if any(s == 'critical' for s in self.component_status.values()):
            summary['overall_status'] = 'critical'
        elif any(s == 'warning' for s in self.component_status.values()):
            summary['overall_status'] = 'warning'
        
        return summary
    
    def generate_alert(self, condition: str, message: str) -> Optional[Dict]:
        """
        Generate alert if condition met
        """
        if condition == 'circuit_breaker':
            return {
                'title': '🚨 Circuit Breaker Triggered',
                'level': 'critical',
                'message': message,
                'timestamp': datetime.now().isoformat()
            }
        
        elif condition == 'high_drawdown':
            return {
                'title': '⚠️ High Drawdown Alert',
                'level': 'warning',
                'message': message,
                'timestamp': datetime.now().isoformat()
            }
        
        elif condition == 'model_degradation':
            return {
                'title': '🔧 Model Performance Degraded',
                'level': 'warning',
                'message': message,
                'timestamp': datetime.now().isoformat()
            }
        
        return None


# Convenience function
def check_system_health() -> Dict:
    """Quick health check"""
    monitor = SystemMonitor()
    return monitor.get_health_summary()
