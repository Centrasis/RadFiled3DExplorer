from dash import dcc
import plotly.graph_objs as go
from typing import Tuple
# important to import kaleido for exporting figures to png, svg, etc.
import kaleido


class PlotFrame:
    def __init__(self, size: Tuple[int, int], key: str = None) -> None:
        self.key = key
        self.size = size
        self.figure = None
        self.graph_component = dcc.Graph(id=key, style={'width': f'{size[0]}px', 'height': f'{size[1]}px'})
        self.background_color = '#2e2e2e'  # Gentle dark grey background
        self.text_color = '#ffffff'  # Light text color

    def draw_figure(self, data, layout):
        azim_elev = None
        if self.figure is not None:
            if 'scene' in self.figure['layout']:
                azim_elev = (self.figure['layout']['scene']['camera']['eye']['x'], 
                             self.figure['layout']['scene']['camera']['eye']['y'])
        
        self.figure = go.Figure(data=data, layout=layout)
        
        if azim_elev is not None:
            self.figure.update_layout(scene_camera_eye=dict(x=azim_elev[0], y=azim_elev[1]))

        self.figure.update_layout(
            template='plotly_dark',
            paper_bgcolor=self.background_color,
            plot_bgcolor=self.background_color,
            font=dict(color=self.text_color)
        )

        self.graph_component.figure = self.figure

    def save_figure(self, to_path: str):
        if self.figure:
            self.figure.write_image(to_path)
