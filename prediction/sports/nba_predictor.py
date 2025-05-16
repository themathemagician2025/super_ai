# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

import logging
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from nba_api.stats.endpoints import playergamelog, commonplayerinfo, teamgamelog
from nba_api.stats.static import players, teams
import time

logger = logging.getLogger(__name__)

class NBAPredictor:
    """NBA player prop prediction model integrated with Super AI decision making."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.player_cache = {}  # Cache for player data
        self.team_cache = {}    # Cache for team data
        self.api_call_delay = 1.0  # Delay between API calls to avoid rate limiting
        
    def _find_player(self, player_name: str) -> Optional[Dict]:
        """Find a player by name."""
        try:
            # Search for active players only
            player_list = players.find_players_by_full_name(player_name)
            if not player_list:
                # Try partial name search
                all_players = players.get_active_players()
                player_list = [p for p in all_players 
                             if player_name.lower() in p['full_name'].lower()]
            
            if player_list:
                return player_list[0]
            return None
            
        except Exception as e:
            logger.error(f"Error finding player {player_name}: {e}")
            return None
    
    def _find_team(self, team_name: str) -> Optional[Dict]:
        """Find a team by name."""
        try:
            team_list = teams.find_teams_by_full_name(team_name)
            if not team_list:
                # Try partial name search
                all_teams = teams.get_teams()
                team_list = [t for t in all_teams 
                           if team_name.lower() in t['full_name'].lower()]
            
            if team_list:
                return team_list[0]
            return None
            
        except Exception as e:
            logger.error(f"Error finding team {team_name}: {e}")
            return None
    
    def _get_player_game_logs(self, player_id: int, n_games: int = 10) -> pd.DataFrame:
        """Get player's recent game logs."""
        try:
            if player_id in self.player_cache:
                return self.player_cache[player_id]
            
            time.sleep(self.api_call_delay)  # Rate limiting
            game_logs = playergamelog.PlayerGameLog(player_id=player_id)
            df = game_logs.get_data_frames()[0]
            
            # Cache the data
            self.player_cache[player_id] = df
            
            return df.head(n_games)
            
        except Exception as e:
            logger.error(f"Error getting game logs for player {player_id}: {e}")
            return pd.DataFrame()
    
    def _get_team_game_logs(self, team_id: int, n_games: int = 10) -> pd.DataFrame:
        """Get team's recent game logs."""
        try:
            if team_id in self.team_cache:
                return self.team_cache[team_id]
            
            time.sleep(self.api_call_delay)  # Rate limiting
            game_logs = teamgamelog.TeamGameLog(team_id=team_id)
            df = game_logs.get_data_frames()[0]
            
            # Cache the data
            self.team_cache[team_id] = df
            
            return df.head(n_games)
            
        except Exception as e:
            logger.error(f"Error getting game logs for team {team_id}: {e}")
            return pd.DataFrame()
    
    def _calculate_player_averages(self, game_logs: pd.DataFrame) -> Dict[str, float]:
        """Calculate player's average statistics."""
        if game_logs.empty:
            return {}
        
        # Calculate basic averages
        averages = {
            'points': game_logs['PTS'].mean(),
            'rebounds': game_logs['REB'].mean(),
            'assists': game_logs['AST'].mean(),
            'steals': game_logs['STL'].mean(),
            'blocks': game_logs['BLK'].mean(),
            'minutes': game_logs['MIN'].mean(),
            'turnovers': game_logs['TOV'].mean(),
            'field_goal_pct': game_logs['FG_PCT'].mean(),
            'three_point_pct': game_logs['FG3_PCT'].mean()
        }
        
        # Calculate combo stats
        averages.update({
            'points_rebounds': averages['points'] + averages['rebounds'],
            'points_assists': averages['points'] + averages['assists'],
            'rebounds_assists': averages['rebounds'] + averages['assists'],
            'points_rebounds_assists': (averages['points'] + 
                                      averages['rebounds'] + 
                                      averages['assists'])
        })
        
        return averages
    
    def _calculate_matchup_factors(
        self,
        player_position: str,
        opponent_team_logs: pd.DataFrame
    ) -> Dict[str, float]:
        """Calculate matchup-specific adjustment factors."""
        if opponent_team_logs.empty:
            return {}
        
        # Calculate team defensive ratings
        defensive_factors = {
            'points': opponent_team_logs['PTS'].mean() / 100,
            'rebounds': opponent_team_logs['REB'].mean() / 40,
            'assists': opponent_team_logs['AST'].mean() / 25,
            'steals': opponent_team_logs['STL'].mean() / 8,
            'blocks': opponent_team_logs['BLK'].mean() / 5
        }
        
        # Adjust based on position
        position_multipliers = {
            'G': {'points': 1.1, 'assists': 1.2, 'rebounds': 0.8},
            'F': {'points': 1.0, 'rebounds': 1.1, 'assists': 0.9},
            'C': {'points': 0.9, 'rebounds': 1.2, 'assists': 0.7}
        }
        
        pos_mult = position_multipliers.get(player_position[0], 
                                          {'points': 1.0, 'rebounds': 1.0, 'assists': 1.0})
        
        return {
            stat: factor * pos_mult.get(stat, 1.0)
            for stat, factor in defensive_factors.items()
        }
    
    def predict_player_props(
        self,
        player_name: str,
        opponent_team: str,
        n_games: int = 10
    ) -> Dict[str, Union[float, Dict]]:
        """Predict player prop statistics for a game."""
        # Find player and team
        player = self._find_player(player_name)
        opponent = self._find_team(opponent_team)
        
        if not player or not opponent:
            return {'error': 'Player or team not found'}
        
        # Get game logs
        player_logs = self._get_player_game_logs(player['id'], n_games)
        opponent_logs = self._get_team_game_logs(opponent['id'], n_games)
        
        if player_logs.empty or opponent_logs.empty:
            return {'error': 'No recent game data available'}
        
        # Get player info for position
        time.sleep(self.api_call_delay)
        player_info = commonplayerinfo.CommonPlayerInfo(player_id=player['id'])
        player_position = player_info.get_data_frames()[0]['POSITION'].iloc[0]
        
        # Calculate base averages
        base_averages = self._calculate_player_averages(player_logs)
        
        # Calculate matchup factors
        matchup_factors = self._calculate_matchup_factors(player_position, opponent_logs)
        
        # Calculate projected stats
        projections = {}
        confidence_scores = {}
        
        for stat, base_avg in base_averages.items():
            # Apply matchup adjustment
            if stat in matchup_factors:
                adj_factor = matchup_factors[stat]
                proj_value = base_avg * adj_factor
            else:
                proj_value = base_avg
            
            # Calculate confidence score (0-1)
            consistency = 1 - (player_logs[stat.upper()].std() / base_avg 
                             if stat.upper() in player_logs else 0.5)
            confidence = min(1.0, max(0.1, consistency))
            
            projections[stat] = proj_value
            confidence_scores[stat] = confidence
        
        return {
            'player_info': {
                'name': player['full_name'],
                'position': player_position,
                'team': player.get('team', 'N/A')
            },
            'projections': projections,
            'confidence_scores': confidence_scores,
            'analysis': {
                'recent_games_analyzed': len(player_logs),
                'average_minutes': base_averages.get('minutes', 0),
                'matchup_factors': matchup_factors
            }
        }
    
    def get_player_form_analysis(
        self,
        player_name: str,
        n_games: int = 5
    ) -> Dict[str, any]:
        """Get detailed form analysis for a player."""
        player = self._find_player(player_name)
        if not player:
            return {'error': f'Player not found: {player_name}'}
        
        game_logs = self._get_player_game_logs(player['id'], n_games)
        if game_logs.empty:
            return {'error': 'No recent game data available'}
        
        # Calculate form metrics
        recent_performances = []
        for _, game in game_logs.iterrows():
            performance = {
                'game_date': game['GAME_DATE'],
                'opponent': game['MATCHUP'],
                'points': game['PTS'],
                'rebounds': game['REB'],
                'assists': game['AST'],
                'minutes': game['MIN'],
                'plus_minus': game['PLUS_MINUS']
            }
            recent_performances.append(performance)
        
        # Calculate trends
        stats_trends = {}
        for stat in ['PTS', 'REB', 'AST']:
            values = game_logs[stat].values
            trend = 'stable'
            if len(values) >= 3:
                if values[0] > values[-1] * 1.1:
                    trend = 'decreasing'
                elif values[0] < values[-1] * 0.9:
                    trend = 'increasing'
            stats_trends[stat.lower()] = trend
        
        return {
            'player_info': player,
            'recent_performances': recent_performances,
            'trends': stats_trends,
            'averages': self._calculate_player_averages(game_logs)
        }
    
    def get_matchup_analysis(
        self,
        player_name: str,
        opponent_team: str
    ) -> Dict[str, any]:
        """Analyze player's historical performance against specific opponent."""
        player = self._find_player(player_name)
        opponent = self._find_team(opponent_team)
        
        if not player or not opponent:
            return {'error': 'Player or team not found'}
        
        # Get all game logs
        game_logs = self._get_player_game_logs(player['id'], 82)  # Full season
        if game_logs.empty:
            return {'error': 'No game data available'}
        
        # Filter games against opponent
        opponent_games = game_logs[game_logs['MATCHUP'].str.contains(opponent['abbreviation'])]
        
        if opponent_games.empty:
            return {'error': 'No games found against this opponent'}
        
        # Calculate stats against opponent
        opponent_stats = self._calculate_player_averages(opponent_games)
        
        # Calculate overall stats for comparison
        overall_stats = self._calculate_player_averages(game_logs)
        
        # Calculate performance differentials
        differentials = {
            stat: opponent_stats.get(stat, 0) - overall_stats.get(stat, 0)
            for stat in opponent_stats.keys()
        }
        
        return {
            'games_against_opponent': len(opponent_games),
            'stats_vs_opponent': opponent_stats,
            'overall_stats': overall_stats,
            'differentials': differentials,
            'recent_matchups': opponent_games[['GAME_DATE', 'MATCHUP', 'PTS', 'REB', 'AST']].to_dict('records')
        } 