import plotly.express as px
from dash import Dash, dcc, html, Input, Output, no_update
from skimage import data
import json

img = data.chelsea()
fig = px.imshow(img)
fig.update_layout(dragmode="drawclosedpath")
config = {
    "modeBarButtonsToAdd": [
        "drawline",
        "drawopenpath",
        "drawclosedpath",
        "drawcircle",
        "drawrect",
        "eraseshape",
    ]
}

# Build App
app = Dash(__name__)
app.layout = html.Div(
    [
        html.H4(
            "Drag and draw annotations - use the modebar to pick a different drawing tool"
        ),
        dcc.Graph(id="graph-pic", figure=fig, config=config),
        dcc.Markdown("Characteristics of shapes"),
        html.Pre(id="annotations-data-pre"),
    ]
)

@app.callback(
    Output("annotations-data-pre", "children"),
    Input("graph-pic", "relayoutData"),
    prevent_initial_call=True,
)
def on_new_annotation(relayout_data):
    if "shapes" in relayout_data:
        return json.dumps(relayout_data["shapes"], indent=2)
    else:
        return no_update

if __name__ == "__main__":
    app.run_server(mode="inline")
