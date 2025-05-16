# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Import actual implementations
from models.lstm_decision import LSTMDecisionMaker
from models.minmax_search import MinMaxSearcher
from models.rational_agent import RationalAgent
from data.manager import DataManager

logger = logging.getLogger(__name__)

class SystemConfiguration:
    """Manages system-wide configuration settings for the AI"""

    def __init__(self, config_path: Path = Path("config/system.yaml")):
        self.config_path = config_path
        self.config = {}

        # System modules configuration
        self.ai_config = {}
        self.data_config = {}
        self.web_config = {}
        self.self_mod_config = {}
        self.ui_config = {}

        # System settings
        self.debug_mode = False
        self.verbose_output = False
        self.save_predictions = True
        self.use_external_apis = False

        # Performance settings
        self.use_parallel = True
        self.use_gpu = False
        self.max_threads = 4
        self.batch_size = 32

        # Load configuration
        self._load_config()
        logger.info("System configuration initialized")

    def _load_config(self):
        """Load configuration from YAML file"""
        try:
            import yaml
            if self.config_path.exists():
                with open(self.config_path, 'r') as file:
                    self.config = yaml.safe_load(file)

                # Apply configuration settings
                if self.config:
                    # System settings
                    if 'system' in self.config:
                        sys_cfg = self.config['system']
                        self.debug_mode = sys_cfg.get('debug_mode', self.debug_mode)
                        self.verbose_output = sys_cfg.get('verbose_output', self.verbose_output)
                        self.save_predictions = sys_cfg.get('save_predictions', self.save_predictions)
                        self.use_external_apis = sys_cfg.get('use_external_apis', self.use_external_apis)

                    # Performance settings
                    if 'performance' in self.config:
                        perf_cfg = self.config['performance']
                        self.use_parallel = perf_cfg.get('use_parallel', self.use_parallel)
                        self.use_gpu = perf_cfg.get('use_gpu', self.use_gpu)
                        self.max_threads = perf_cfg.get('max_threads', self.max_threads)
                        self.batch_size = perf_cfg.get('batch_size', self.batch_size)

                    # Module configurations
                    if 'ai' in self.config:
                        self.ai_config = self.config['ai']

                    if 'data' in self.config:
                        self.data_config = self.config['data']

                    if 'web' in self.config:
                        self.web_config = self.config['web']

                    if 'self_modification' in self.config:
                        self.self_mod_config = self.config['self_modification']

                    if 'ui' in self.config:
                        self.ui_config = self.config['ui']

                logger.info(f"Loaded configuration from {self.config_path}")
            else:
                logger.warning(f"Configuration file {self.config_path} not found, using defaults")
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")

    def get(self, section: str, key: str, default: Any = None) -> Any:
        """Get a configuration value with fallback to default"""
        try:
            if section in self.config and key in self.config[section]:
                return self.config[section][key]
            return default
        except Exception as e:
            logger.error(f"Error getting configuration {section}.{key}: {str(e)}")
            return default

    def set(self, section: str, key: str, value: Any) -> bool:
        """Set a configuration value"""
        try:
            if section not in self.config:
                self.config[section] = {}

            self.config[section][key] = value
            logger.debug(f"Set configuration {section}.{key} = {value}")
            return True
        except Exception as e:
            logger.error(f"Error setting configuration {section}.{key}: {str(e)}")
            return False

    def save(self) -> bool:
        """Save the current configuration to file"""
        try:
            import yaml
            with open(self.config_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False)

            logger.info(f"Saved configuration to {self.config_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")
            return False

    def is_module_enabled(self, module_name: str) -> bool:
        """Check if a module is enabled in the configuration"""
        return self.get('modules', module_name, True)

    def get_api_key(self, service_name: str) -> Optional[str]:
        """Get an API key for a service"""
        return self.get('api_keys', service_name, None)

    def get_module_config(self, module_name: str) -> Dict[str, Any]:
        """Get the complete configuration for a module"""
        if module_name in self.config:
            return self.config[module_name]
        return {}

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information and settings"""
        return {
            'debug_mode': self.debug_mode,
            'verbose_output': self.verbose_output,
            'save_predictions': self.save_predictions,
            'use_external_apis': self.use_external_apis,
            'use_parallel': self.use_parallel,
            'use_gpu': self.use_gpu,
            'max_threads': self.max_threads,
            'batch_size': self.batch_size,
            'config_path': str(self.config_path)
        }

class SystemIntegrator:
    """Integrates AI components for the Mathemagician system"""

    def __init__(self, config: Dict[str, Any]):
        """Initialize system components"""
        self.config = config
        try:
            self.decision_maker = LSTMDecisionMaker()
            self.searcher = MinMaxSearcher()
            self.agent = RationalAgent()
            self.data_manager = DataManager()
            logger.info("System integrator initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize system integrator: {str(e)}")
            raise

    def analyze_market(self) -> Dict[str, Any]:
        """Analyze market patterns using LSTM"""
        try:
            market_data = self.data_manager.get_market_data()
            analysis = self.decision_maker.analyze_patterns(market_data)
            logger.info("Market analysis completed")
            return analysis
        except AttributeError:
            logger.error("LSTMDecisionMaker missing required method")
            return {"error": "Analysis method not available"}
        except Exception as e:
            logger.error(f"Market analysis failed: {str(e)}")
            return {"error": str(e)}

    def analyze_game(self) -> Dict[str, Any]:
        """Analyze game state using MinMax search"""
        try:
            game_state = self.data_manager.get_game_state()
            analysis = self.searcher.analyze_game_state(game_state)
            logger.info("Game analysis completed")
            return analysis
        except AttributeError:
            logger.error("MinMaxSearcher missing required method")
            return {"error": "Analysis method not available"}
        except Exception as e:
            logger.error(f"Game analysis failed: {str(e)}")
            return {"error": str(e)}

    def make_decision(self) -> Dict[str, Any]:
        """Generate decision using rational agent"""
        try:
            context = self.data_manager.get_context()
            decision = self.agent.make_decision(context)
            logger.info("Decision making completed")
            return decision
        except AttributeError:
            logger.error("RationalAgent missing required method")
            return {"error": "Decision method not available"}
        except Exception as e:
            logger.error(f"Decision making failed: {str(e)}")
            return {"error": str(e)}

    def full_analysis(self) -> Dict[str, Any]:
        """Perform comprehensive analysis"""
        try:
            results = {
                "market": self.analyze_market(),
                "game": self.analyze_game(),
                "decision": self.make_decision()
            }
            logger.info("Full analysis completed")
            return results
        except Exception as e:
            logger.error(f"Full analysis failed: {str(e)}")
            return {"error": str(e)}

    def run(self) -> Dict[str, Any]:
        """Run the complete analysis pipeline"""
        try:
            results = self.full_analysis()
            if "error" in results:
                logger.warning("Analysis completed with errors")
            else:
                logger.info("Analysis pipeline completed successfully")
            return results
        except Exception as e:
            logger.error(f"Analysis pipeline failed: {str(e)}")
            return {"error": str(e)}
