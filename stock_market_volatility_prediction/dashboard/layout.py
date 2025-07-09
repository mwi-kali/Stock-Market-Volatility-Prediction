import dash_bootstrap_components as dbc


from dash import html, dcc


def create_layout():
    cards = dbc.CardGroup(
        [
            dbc.Card(
                [
                    dbc.CardHeader("Latest Actual"),
                    dbc.CardBody(html.H4(id="card-actual", className="card-title")),
                ],
                color="light",
            ),
            dbc.Card(
                [
                    dbc.CardHeader("LSTM Pred."),
                    dbc.CardBody(html.H4(id="card-lstm", className="card-title")),
                ],
                color="info",
                inverse=True,
            ),
            dbc.Card(
                [
                    dbc.CardHeader("ARIMA Pred."),
                    dbc.CardBody(html.H4(id="card-arima", className="card-title")),
                ],
                color="success",
                inverse=True,
            ),
            dbc.Card(
                [
                    dbc.CardHeader("Prophet Pred."),
                    dbc.CardBody(html.H4(id="card-prophet", className="card-title")),
                ],
                color="warning",
                inverse=True,
            ),
        ],
        className="mb-4",
    )

    graph_card = dbc.Card(
        [
            dbc.CardHeader(html.H5("Actual Stock Volatility vs. Predicted Stock Volatility")),
            dbc.CardBody(
                dcc.Loading(dcc.Graph(id="forecast-graph"), type="circle"),
                style={"height": "600px"},
            ),
        ],
        className="mb-4",
    )

    layout = dbc.Container(
        [
            dbc.Row(
                dbc.Col(
                    html.H1("Stock Volatility Forecast Dashboard"),
                    width=12,
                ),
                className="my-4",
            ),
            dbc.Row(
                dbc.Col(cards, width=12),
                className="mb-4",
            ),
            dbc.Row(
                dbc.Col(graph_card, width=12),
            ),
            dcc.Interval(id="interval", interval=1, n_intervals=0, max_intervals=1),
        ],
        fluid=True,
    )

    return layout
