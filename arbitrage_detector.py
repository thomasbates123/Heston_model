"""
Options Arbitrage Detection Module

This module implements various arbitrage detection strategies for options markets.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple


class ArbitrageDetector:
    """
    Detect various types of arbitrage opportunities in options markets.
    """
    
    def __init__(self, options_df, stock_price, risk_free_rate=0.05, tolerance=0.01):
        """
        Initialize arbitrage detector.
        
        Parameters:
        -----------
        options_df : pd.DataFrame
            Options data with columns: date, strike_price, T, cp_flag, mid_price, best_bid, best_offer
        stock_price : float
            Current underlying stock price
        risk_free_rate : float
            Risk-free interest rate (annual)
        tolerance : float
            Minimum profit threshold to consider an arbitrage (to account for transaction costs)
        """
        self.options_df = options_df
        self.stock_price = stock_price
        self.r = risk_free_rate
        self.tolerance = tolerance
        
    def detect_put_call_parity_violations(self):
        """
        Detect put-call parity violations.
        
        Put-Call Parity: C - P = S - K * e^(-rT)
        
        Returns:
        --------
        pd.DataFrame
            Detected arbitrage opportunities with expected profit
        """
        arbitrage_opportunities = []
        
        # Group by date, expiry, and strike
        grouped = self.options_df.groupby(['date', 'exdate', 'strike_price', 'T'])
        
        for (date, exdate, strike, T), group in grouped:
            calls = group[group['cp_flag'] == 'C']
            puts = group[group['cp_flag'] == 'P']
            
            if len(calls) == 0 or len(puts) == 0:
                continue
                
            call_price = calls['mid_price'].iloc[0]
            put_price = puts['mid_price'].iloc[0]
            call_bid = calls['best_bid'].iloc[0]
            call_offer = calls['best_offer'].iloc[0]
            put_bid = puts['best_bid'].iloc[0]
            put_offer = puts['best_offer'].iloc[0]
            
            # Theoretical relationship
            parity_value = self.stock_price - strike * np.exp(-self.r * T)
            actual_difference = call_price - put_price
            
            violation = abs(actual_difference - parity_value)
            
            # Check for arbitrage considering bid-ask spread
            # Strategy 1: Buy call, sell put (if call - put < S - Ke^-rT)
            buy_call_sell_put_cost = call_offer - put_bid
            if parity_value - buy_call_sell_put_cost > self.tolerance:
                arbitrage_opportunities.append({
                    'date': date,
                    'expiry': exdate,
                    'strike': strike,
                    'T': T,
                    'type': 'Put-Call Parity',
                    'strategy': 'Buy Call, Sell Put, Short Stock, Lend PV(K)',
                    'expected_profit': parity_value - buy_call_sell_put_cost,
                    'call_price': call_price,
                    'put_price': put_price,
                    'violation': violation
                })
            
            # Strategy 2: Sell call, buy put (if call - put > S - Ke^-rT)
            sell_call_buy_put_revenue = call_bid - put_offer
            if sell_call_buy_put_revenue - parity_value > self.tolerance:
                arbitrage_opportunities.append({
                    'date': date,
                    'expiry': exdate,
                    'strike': strike,
                    'T': T,
                    'type': 'Put-Call Parity',
                    'strategy': 'Sell Call, Buy Put, Buy Stock, Borrow PV(K)',
                    'expected_profit': sell_call_buy_put_revenue - parity_value,
                    'call_price': call_price,
                    'put_price': put_price,
                    'violation': violation
                })
        
        return pd.DataFrame(arbitrage_opportunities)
    
    def detect_butterfly_arbitrage(self):
        """
        Detect butterfly spread arbitrage opportunities.
        
        A butterfly spread involves three strikes: K1 < K2 < K3
        where K2 = (K1 + K3) / 2
        
        No arbitrage condition: C(K1) - 2*C(K2) + C(K3) >= 0
        
        Returns:
        --------
        pd.DataFrame
            Detected arbitrage opportunities
        """
        arbitrage_opportunities = []
        
        # Group by date and expiry
        grouped = self.options_df.groupby(['date', 'exdate', 'T'])
        
        for (date, exdate, T), group in grouped:
            calls = group[group['cp_flag'] == 'C'].sort_values('strike_price')
            
            if len(calls) < 3:
                continue
            
            strikes = calls['strike_price'].values
            prices = calls['mid_price'].values
            bids = calls['best_bid'].values
            offers = calls['best_offer'].values
            
            # Check all possible butterfly combinations
            for i in range(len(strikes) - 2):
                for j in range(i + 1, len(strikes) - 1):
                    for k in range(j + 1, len(strikes)):
                        K1, K2, K3 = strikes[i], strikes[j], strikes[k]
                        
                        # Check if strikes are approximately equally spaced
                        if abs((K1 + K3) / 2 - K2) > 0.1:
                            continue
                        
                        # Calculate butterfly value using mid prices
                        butterfly_value = prices[i] - 2 * prices[j] + prices[k]
                        
                        # Check with bid-ask spread
                        # Cost to establish butterfly
                        cost = offers[i] - 2 * bids[j] + offers[k]
                        
                        if cost < -self.tolerance:  # Negative cost means we receive premium
                            arbitrage_opportunities.append({
                                'date': date,
                                'expiry': exdate,
                                'T': T,
                                'type': 'Butterfly Spread',
                                'strategy': f'Buy {K1} call, Sell 2x {K2} calls, Buy {K3} call',
                                'strikes': f'{K1}/{K2}/{K3}',
                                'expected_profit': -cost,
                                'butterfly_value': butterfly_value
                            })
        
        return pd.DataFrame(arbitrage_opportunities)
    
    def detect_calendar_arbitrage(self):
        """
        Detect calendar spread arbitrage opportunities.
        
        For the same strike, options with longer maturity should not be cheaper
        than options with shorter maturity (for calls out of the money).
        
        Returns:
        --------
        pd.DataFrame
            Detected arbitrage opportunities
        """
        arbitrage_opportunities = []
        
        # Group by date, strike, and option type
        grouped = self.options_df.groupby(['date', 'strike_price', 'cp_flag'])
        
        for (date, strike, cp_flag), group in grouped:
            if len(group) < 2:
                continue
            
            # Sort by maturity
            sorted_group = group.sort_values('T')
            
            for i in range(len(sorted_group) - 1):
                short_term = sorted_group.iloc[i]
                long_term = sorted_group.iloc[i + 1]
                
                # Long-term option should be more expensive
                if long_term['mid_price'] < short_term['mid_price'] - self.tolerance:
                    profit = short_term['best_bid'] - long_term['best_offer']
                    
                    if profit > self.tolerance:
                        arbitrage_opportunities.append({
                            'date': date,
                            'strike': strike,
                            'type': 'Calendar Spread',
                            'option_type': 'Call' if cp_flag == 'C' else 'Put',
                            'strategy': f'Sell {short_term["T"]:.2f}y, Buy {long_term["T"]:.2f}y',
                            'short_term_T': short_term['T'],
                            'long_term_T': long_term['T'],
                            'short_term_price': short_term['mid_price'],
                            'long_term_price': long_term['mid_price'],
                            'expected_profit': profit
                        })
        
        return pd.DataFrame(arbitrage_opportunities)
    
    def detect_box_spread_arbitrage(self):
        """
        Detect box spread arbitrage opportunities.
        
        A box spread consists of a bull call spread and a bear put spread.
        Box Spread Value = (K2 - K1) * e^(-rT)
        
        Returns:
        --------
        pd.DataFrame
            Detected arbitrage opportunities
        """
        arbitrage_opportunities = []
        
        # Group by date and expiry
        grouped = self.options_df.groupby(['date', 'exdate', 'T'])
        
        for (date, exdate, T), group in grouped:
            calls = group[group['cp_flag'] == 'C'].sort_values('strike_price')
            puts = group[group['cp_flag'] == 'P'].sort_values('strike_price')
            
            if len(calls) < 2 or len(puts) < 2:
                continue
            
            # Check pairs of strikes
            for i in range(len(calls) - 1):
                K1 = calls.iloc[i]['strike_price']
                K2 = calls.iloc[i + 1]['strike_price']
                
                # Find corresponding puts
                put1 = puts[puts['strike_price'] == K1]
                put2 = puts[puts['strike_price'] == K2]
                
                if len(put1) == 0 or len(put2) == 0:
                    continue
                
                # Box spread = (C(K1) - C(K2)) - (P(K1) - P(K2))
                call_spread_cost = calls.iloc[i]['best_offer'] - calls.iloc[i + 1]['best_bid']
                put_spread_revenue = put1.iloc[0]['best_bid'] - put2.iloc[0]['best_offer']
                
                box_cost = call_spread_cost - put_spread_revenue
                box_payoff = (K2 - K1) * np.exp(-self.r * T)
                
                profit = box_payoff - box_cost
                
                if profit > self.tolerance:
                    arbitrage_opportunities.append({
                        'date': date,
                        'expiry': exdate,
                        'T': T,
                        'type': 'Box Spread',
                        'strikes': f'{K1}/{K2}',
                        'strategy': f'Buy {K1} call, Sell {K2} call, Sell {K1} put, Buy {K2} put',
                        'box_cost': box_cost,
                        'box_payoff': box_payoff,
                        'expected_profit': profit
                    })
        
        return pd.DataFrame(arbitrage_opportunities)
    
    def detect_all_arbitrage(self):
        """
        Run all arbitrage detection methods and return combined results.
        
        Returns:
        --------
        dict
            Dictionary with all detected arbitrage opportunities by type
        """
        results = {
            'put_call_parity': self.detect_put_call_parity_violations(),
            'butterfly': self.detect_butterfly_arbitrage(),
            'calendar': self.detect_calendar_arbitrage(),
            'box_spread': self.detect_box_spread_arbitrage()
        }
        
        # Print summary
        print("\n" + "="*70)
        print("ARBITRAGE DETECTION SUMMARY")
        print("="*70)
        
        total_opportunities = 0
        for arb_type, df in results.items():
            count = len(df)
            total_opportunities += count
            print(f"{arb_type.replace('_', ' ').title()}: {count} opportunities")
            if count > 0 and 'expected_profit' in df.columns:
                print(f"  Total Expected Profit: ${df['expected_profit'].sum():.2f}")
                print(f"  Average Profit: ${df['expected_profit'].mean():.2f}")
        
        print(f"\nTotal Arbitrage Opportunities: {total_opportunities}")
        print("="*70 + "\n")
        
        return results
