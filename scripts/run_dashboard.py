import flask
import stock_market_volatility_prediction.dashboard.callbacks


import dash_bootstrap_components as dbc


from dash import Dash
from stock_market_volatility_prediction.dashboard.cache import cache
from stock_market_volatility_prediction.dashboard.layout import create_layout


server = flask.Flask(__name__)
cache.init_app(server)

external_stylesheets = [dbc.themes.LUX]

app = Dash(__name__, server=server, external_stylesheets=external_stylesheets)
app.title = "Stock Volatility Forecast"
app.layout = create_layout()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050)
