# Stock Market Volatility Prediction

A lightweight, modular pipeline for forecasting stock market volatility by combining technical indicators, news sentiment, and macroeconomic factors.

## Project Structure

```
├── data/
│   ├── raw/             
│   └── processed/      
├── scripts/              
│   ├── run_ingest.py     
│   ├── run_features.py   
│   ├── run_train.py     
│   ├── run_backtest.py   
│   ├── run_dashboard.py  
│   └── run_experiment.py  
├── stock_market_volatility_prediction/
│   ├── ingestion/        
│   ├── features/          
│   ├── models/          
│   ├── dashboard/       
│   └── utils/            
├── models/           
├── setup.py    
├── requirements.txt 
└── README.md              
```

## Quick Start

1. **Install**

   ```bash
   git clone https://github.com/mwi-kali/Stock-Market-Volatility-Prediction.git
   cd Stock-Market-Volatility-Prediction
   pip install -r requirements.txt
   ```

2. **Ingest data**

   ```bash
   python scripts/run_ingest.py --start 2024-01-01 
   ```

3. **Build features**

   ```bash
   python scripts/run_features.py 
   ```

4. **Tune & train**

   ```bash
   # Tune hyperparameters
   python scripts/run_train.py --mode tune --window 10

   # Train models with tuned params
   python scripts/run_train.py --mode train --window 10
   ```

5. **Backtest**

   ```bash
   python scripts/run_backtest.py --initial 100 --step 10
   ```

6. **Dashboard**

   ```bash
   python scripts/run_dashboard.py
   # Open http://127.0.0.1:8050
   ```

7. **Full experiment** (with MLflow)

   ```bash
   mlflow ui          
   python scripts/run_experiment.py --start 2024-01-01 --window 10 --tune-trials 20
   ```
