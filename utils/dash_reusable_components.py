from textwrap import dedent

import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html

# Display utility functions
def _merge(a, b):
    return dict(a, **b)


def _omit(omitted_keys, d):
    return {k: v for k, v in d.items() if k not in omitted_keys}


# Custom Display Components
def Card(children, **kwargs):
    return html.Section(className="card", children=children, **_omit(["style"], kwargs))


def FormattedSlider(**kwargs):
    return html.Div(
        style=kwargs.get("style", {}), children=dcc.Slider(**_omit(["style"], kwargs))
    )


def NamedSlider(name, **kwargs):
    return html.Div(
        style={"padding": "20px 10px 25px 4px"},
        children=[
            html.P(f"{name}:"),
            html.Div(style={"margin-left": "6px"}, children=dcc.Slider(**kwargs)),
        ],
    )


def NamedTextbox(name, **kwargs):
    style = {"color": "white"}
    return html.Div(
        style={"padding": "20px 10px 25px 4px"},
        children=[
            html.P(f"{name}:"),
            html.Div(children=dcc.Input(style=style, **kwargs)),
        ],
    )

def CardComponent(name, **kwargs): 
    card = dbc.Card(
        [
            dbc.CardHeader("This is the header"),
            dbc.CardBody(
                [
                    html.H4("Card title", className="card-title"),
                    html.P("This is some card text", className="card-text"),
                ]
            ),
            dbc.CardFooter("This is the footer"),
        ],
        style={"width": "18rem"},
        color="yellow", inverse=True,
    )
    return card

def Cards(titles, descriptions, divId):
    card_content = [
    dbc.CardHeader("Card header"),
    dbc.CardBody(
            [
                html.H5("Card title", className="card-title"),
                html.P(
                    "This is some card content that we'll reuse",
                    className="card-text",
                ),
            ]
        ),
    ]

    cards = html.Div(
        [
            dbc.Row(
                [
                    dbc.Col(dbc.Card(card_content, color="primary", inverse=True)),
                    dbc.Col(
                        dbc.Card(card_content, color="secondary", inverse=True)
                    ),
                    dbc.Col(dbc.Card(card_content, color="info", inverse=True)),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Card(card_content, color="success", inverse=True)),
                    dbc.Col(dbc.Card(card_content, color="warning", inverse=True)),
                    dbc.Col(dbc.Card(card_content, color="danger", inverse=True)),
                ],
                className="mb-4",
            ),
            dbc.Row(
                [
                    dbc.Col(dbc.Card(card_content, color="light")),
                    dbc.Col(dbc.Card(card_content, color="dark", inverse=True)),
                ]
            ),
        ],
        id=divId,
        style={"width": "67%"},
    )
    return cards


def NamedDropdown(name, **kwargs):
    return html.Div(
        style={"margin": "10px 0px"},
        children=[
            html.P(children=f"{name}:", style={"margin-left": "3px"}),
            dcc.Dropdown(**kwargs),
        ],
    )


def NamedRadioItems(name, **kwargs):
    return html.Div(
        style={"padding": "20px 10px 25px 4px"},
        children=[html.P(children=f"{name}:"), dcc.RadioItems(**kwargs)],
    )

def UnNamedRadioItems(**kwargs):
    return html.Div(
        style={"padding": "20px 10px 25px 4px"},
        children=[dcc.RadioItems(**kwargs)],
    )


# Non-generic
def DemoDescription(filename, strip=False):
    with open(filename, "r") as file:
        text = file.read()

    if strip:
        text = text.split("<Start Description>")[-1]
        text = text.split("<End Description>")[0]

    return html.Div(
        className="row",
        style={
            "padding": "15px 30px 27px",
            "margin": "45px auto 45px",
            "width": "80%",
            "max-width": "1024px",
            "borderRadius": 5,
            "border": "thin lightgrey solid",
            "font-family": "Roboto, sans-serif",
        },
        children=dcc.Markdown(dedent(text)),
    )