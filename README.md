# Trader Sentiment Analysis (Hyperliquid + Bitcoin Fear & Greed)

## 1. Objective

Analyze how Hyperliquid traders perform under different **Bitcoin market sentiment regimes**:

- Extreme Fear  
- Fear  
- Neutral  
- Greed  
- Extreme Greed  

The goal is to understand how PnL, win rate, and trade behavior change across these regimes and derive insights that can help design smarter trading strategies.

---

## 2. Data

### 2.1 Hyperliquid Historical Trader Data  
Each row = one executed trade.

Key columns used:
- `Account` – trader wallet / account id  
- `Coin` – traded symbol  
- `Execution Price`  
- `Size Tokens`, `Size USD`  
- `Side` – BUY / SELL  
- `Timestamp IST` – trade time (used to derive trade date)  
- `Closed PnL` – PnL for that trade  

### 2.2 Bitcoin Fear & Greed Index  
Columns:
- `date` – calendar date (used to align with trades)  
- `classification` – {Extreme Fear, Fear, Neutral, Greed, Extreme Greed}  
- `value`, `timestamp` – index metadata  

We join trades with sentiment using **date**.

---

## 3. Methodology

### 3.1 Preprocessing

1. **Trades**
   - Parse `Timestamp IST` → `datetime`.
   - Create `trade_date = date(Timestamp IST)`.

2. **Sentiment**
   - Parse `date` → `datetime`.
   - Create `sentiment_date = date(date)`.
   - Use `classification` as the market sentiment label.

### 3.2 Merging

Left join:

```text
(trades) trade_date  ==  sentiment_date (Fear & Greed)
