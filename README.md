# Quantitative Investment Strategy – MSc Thesis

This repository contains the Python implementation of my MSc Finance thesis at ESADE:

**“Does LLM-based qualitative analysis improve the risk-adjusted returns of momentum-driven portfolios?”**

---

## Overview

This project develops and backtests a systematic equity strategy on the S&P 500 combining:

- Momentum signals  
- LLM-based sentiment analysis (Reddit + Yahoo Finance news)  

The objective is to enhance risk-adjusted returns by integrating qualitative information into a quantitative investment framework.

---

## Key Results

- +60–70 bps annual alpha vs baseline momentum strategy
- Sharpe ratio improvement (~0.40 → ~0.44)  
- Comparable volatility with improved risk-adjusted returns  

---

## Methodology

The strategy combines multiple components:

### 1. Data Collection
- Reddit sentiment (r/stocks, r/investing, r/wallstreetbets)  
- Yahoo Finance news headlines  

### 2. NLP Pipeline
- **FinBERT** → sentiment analysis  
- **BART-MNLI (zero-shot)** → risk detection  
- Aggregated into a composite sentiment-risk signal  

### 3. Portfolio Construction
- Momentum-based stock selection (top S&P 500 performers)  
- Sentiment-based tilting (over/underweighting)  
- Monthly rebalancing  

### 4. Optimization
- Tracking-error constrained portfolio  
- Turnover control  
- Risk management overlay  

---

## Code Structure

- `Thesis_Code_V12.py` → **Main strategy (portfolio construction, rebalancing, backtest)**  
- `Portfolio_Return_Calculator.py` → Return computation  
- `Portfolio_Return_Calculator_Opt.py` → Optimized portfolio returns  
- `Rolling_Factor_Regression_V2.py` → Factor analysis (Carhart / Fama-French)  
- `Rolling_Factor_Regression_V2_Opt.py` → Factor analysis for optimized portfolio  
- `Text_Retriever.py` → Fetches Reddit + Yahoo Finance data  

---

## Tools & Technologies

- Python (Pandas, NumPy)  
- NLP models (FinBERT, BART-MNLI)  
- Quantitative finance techniques  
- Backtesting & factor regression  

---

## Key Contribution

This project demonstrates that **LLM-based qualitative signals can complement traditional factor investing**, providing:

- Incremental alpha  
- Improved risk-adjusted performance  
- Enhanced portfolio robustness  
