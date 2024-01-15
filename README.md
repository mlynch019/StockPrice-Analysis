# Stock-Price-Analysis
The focus of this project was to learn management of a SQLite database and to process the data through the creation of tables and executing queries. Data used in this project can be found [here.](https://finance.yahoo.com/quote/GOOG?p=GOOG&.tsrc=fin-srch) The first half of the project involves a sequence of queries to calculate the 5-day, 20-day, 50-day, and 150-day moving averages for the stock, as well as the 14-day Relative Strength Index (RSI), yearly high/low, maximum/minimum/average daily percent changes, and the moving average convergence/divergence indicator (MACD). The contents of these queries were then graphed and can be found in the `SQL_graphs` directory. The second half of the project involved the use of the Keras Sequential class with multiple Long Short Term Memory layers (LSTM) with ReLU activation and a single dense layer in order to predict the stock price over the latter 25% of the 3-year span of data. The model was tested with different optimizers (Adam, Adagrad, Stochastic Gradient Descent (SGD)) and the visual results of the 3 optimizers are provided in the `optimizer_graphs` directory. The loss (mean squared error) of each optimizer was plotted to compare. 
## Preview Images
Sample graphs (all can be found in the `SQL_graphs` and `optimizer_graphs` directories).
<img width="1374" alt="20 Day Moving Average copy" src="https://github.com/mlynch019/Stock-Price-Analysis/assets/113787390/1493dd64-86fc-49b6-b543-015908858fe4">
<img width="1355" alt="ADAM Optimizer copy" src="https://github.com/mlynch019/Stock-Price-Analysis/assets/113787390/7abecc8d-d208-44fd-a91b-e5cdaaefa1bb">

### Sources/Referenced Information

https://www.investopedia.com/ask/answers/122414/what-moving-average-convergence-divergence-macd-formula-and-how-it-calculated.asp

https://keras.io/guides/sequential_model/ 

https://medium.com/@zahmed333/stock-price-prediction-with-keras-df87b05e5906

https://finance.yahoo.com/quote/GOOG?p=GOOG&.tsrc=fin-srch 
