# Stock Prediction ML model 
- Code for used for testing and training various models for predicting stock returns
- Includes code for: downloading datasets using yfinance, computing technical indicators using TALib, a walk forward validation script with inner time series split hyper parameter tuning loop using Optuna; training and evaluation code using Pytorch. And backtesting using backtesting.py and BT
- Most of the models/ methods don't work but the cross-sectional multi ETF model does alright - predicts for each ETF, the cross sectional ranking of the forward return then the strategy picks the top e.g 5 ETFs to buy every week or month
- The full walk forward validation with hyper parameter tuning loop takes quite long to run (since it trains hundreds of models) but running it once shows that the model outperforms the average performance of randomly choosing ETFs to invest in
- However, for actual statistical significance, one would have to run the walk forward validation script for many different random seeds and see if the average performance is better than the random strategy which I unfortunately don't have the time to do
- The code is quite messy as it contains lots of failed strategies, to test the strategy above run multi_ticker_script.walk_forward_validation() then multi_ticker_script.backtest_full() from the main function,  set baseline to true for backtest_full() if you want to see the performance relative to random strategies and false if you just want the equity curve

