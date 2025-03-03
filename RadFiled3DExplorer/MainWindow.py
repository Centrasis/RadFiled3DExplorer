import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
from .PlotFrame import PlotFrame
from .RadSimPlotter import RadSimPlotter, PlotInformation, Noop, FieldComponent
from typing import Tuple, Deque, Callable
from collections import deque
import os
from threading import Thread
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from ptbDataLab.normalizations.log_norm import LogNorm
from ptbDataLab.normalizations.linear_norm import LinearNorm
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import webview
import tkinter as tk
from tkinter import filedialog
from RadFiled3D.RadFiled3D import FieldStore, RadiationFieldMetadataHeaderV1
import numpy as np


class MainWindow(FileSystemEventHandler):
    instance = None
    file_parsing_progress = 0

    @staticmethod
    def on_any_event(event):
        if event.is_directory:
            return
        if event.event_type == 'created' or event.event_type == 'modified':
            file: str = os.path.basename(event.src_path)
            if file.endswith(".rf3"):
                MainWindow.instance.on_new_file_in_dataset(file)
        if event.event_type == 'deleted' or event.event_type == 'removed':
            file = os.path.basename(event.src_path)
            if file.endswith(".rf3"):
                MainWindow.instance.on_file_removed_from_dataset(file)

    def __init__(self, caption: str, size: Tuple[int, int] = (1024, 700)):
        MainWindow.instance = self
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
        self.viewer_plot = PlotFrame(key='-VIEWER-PLOT-', size=(int(size[0] * 0.8), int(size[1] * 0.8)))
        self.stats_plot = PlotFrame(key='-STATS-PLOT-', size=(int(size[0] * 0.65), int(size[1] * 0.65)))
        self.gui_calls: Deque[Callable[[], None]] = deque()
        self.caption_lbl = html.Div('Plot: ')
        self.file_list = []
        self.file_list_box = dcc.Dropdown(id='file-list', options=[], multi=True, style={'height': '200px'})
        self.plotter = RadSimPlotter()
        self.plotter.plot_infos = PlotInformation.Doserate
        self.show_radiation_source = True
        self.dir_watcher: Observer = None
        self.file_idx = 0
        self.norm = Noop()
        self.menu_def = [
            ['Dataset', ['Open', ]],
            ['Viewer', ['Export', ]]
        ]
        self.tabs = html.Div([
            dcc.Tabs(id='tabs', children=[
                dcc.Tab(label='Viewer Plot', children=[
                    self.viewer_plot.graph_component
                ]),
                dcc.Tab(label='Stats Plot', children=[
                    self.stats_plot.graph_component
                ])
            ])
        ])

        self.last_directories = set()
        self.dataset_path = os.path.abspath(os.getcwd())  # Default path

        MainWindow.file_parsing_progress = 100
        MainWindow.last_statistics_calculated = ""

        self.app.layout = html.Div([
            html.Div([ # Sidebar with file explorer
                dcc.Tabs(id='explorer-tabs', value='files', children=[
                    dcc.Tab(label='Files', value='files', style={'height': '7vh'}),
                    dcc.Tab(label='Statistics', value='statistics', style={'height': '7vh'})
                ], style={'height': '7vh'}),
                html.Div(id='explorer-content')
            ], style={'width': '30%', 'height': '100%', 'float': 'left', 'padding': '10px'}),
            html.Div([  # Content of the data display area
                html.Div(id='data-display-area')
            ], style={'width': '70%', 'height': '100%', 'float': 'right', 'padding': '10px'}),
            dcc.Store(id='window-initialized', data=False),
            # JavaScript snippet to trigger a Dash callback when the window is fully loaded
            html.Script('''
                document.addEventListener('DOMContentLoaded', function() {
                    var event = new CustomEvent('window-initialized');
                    window.dispatchEvent(event);
                });
            ''')
        ], style={'height': '100vh', 'display': 'flex'})

        @self.app.callback(
            Output('explorer-content', 'children'),
            [Input('explorer-tabs', 'value')]
        )
        def update_explorer_content(tab):
            if tab == 'files':
                return html.Div([
                    html.H3("File Explorer"),
                    html.Div([
                        dcc.Dropdown(id='directory-list', options=list(self.last_directories), multi=False, placeholder=f"{self.dataset_path}", style={'height': '90%', 'width': '90%'}),
                        html.Button('...', id='select-directory-btn', style={'width': '10%', 'top': '0%', 'right': '0%', 'position': 'relative'})
                    ], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'height': '10vh', 'width': '100%'}),
                    html.Ul(id='file-list', style={'height': '75vh', 'max-height': "75vh", 'overflowY': 'scroll', 'listStyleType': 'none', 'padding': '0', 'margin': '0'}),
                    html.Div([
                        html.Button('<--', id='prev-btn', style={'width': '45%', 'margin-right': '5%'}),
                        html.Button('-->', id='next-btn', style={'width': '45%'})
                    ], style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'height': '10%', 'margin-bottom': '5%', 'bottom': '5%', 'position': 'relative', 'width': '95%'}),
                ], style={'height': '90vh', 'width': '100%'})
            elif tab == 'statistics':
                return html.Div([
                    html.H3("Statistics"),
                    dcc.RadioItems(id='statistics-list',
                        options=[{'label': 'Energy', 'value': 'energy'}, {'label': 'Angles', 'value': 'angles'}, {'label': 'Source Distance', 'value': 'source_distance'}],
                        value='energy',
                        style={'height': '10vh', 'listStyleType': 'none', 'padding': '0', 'margin': '0'}),
                ])

        @self.app.callback(
            Output('data-display-area', 'children'),
            [Input('explorer-tabs', 'value')]
        )
        def update_data_display_area(tab):
            if tab == 'files':
                return html.Div([
                    dcc.RadioItems(
                        id='layer-selection',
                        options=[
                            {'label': 'Air Kerma', 'value': 'Kerma'},
                            {'label': 'Hits', 'value': 'Hits'},
                            {'label': 'Directions', 'value': 'Directions'},
                            {'label': 'Errors', 'value': 'Errors'}
                        ],
                        value='Kerma',
                        labelStyle={'display': 'inline-block', 'margin-right': '10px'},
                        style={'width': '100%', 'justify-content': 'center', 'align-items': 'center', 'display': 'flex', 'height': '5vh'}
                    ),
                    dcc.Loading(
                        html.Div([
                            dcc.Graph(id='viewer-graph', style={'width': 'calc(100% - 50px)', 'height': 'inherit', 'float': 'left'}),
                            html.Div([
                                dcc.Slider(id='vertical-slider', vertical=True, min=0, max=100, step=1, value=50, marks={0: '0', 100: '100'})
                            ], style={'width': '50px', 'height': '80vh', 'float': 'right', 'display': 'flex', "justify-content": "center", "align-items": "center"})
                        ], style={'width': '100%', 'height': '80vh', 'display': 'flex', "justify-content": "center", "align-items": "center"}),
                        id="loading",
                        type="default",
                        style={'width': '100%', 'height': '80vh', 'display': 'flex', "justify-content": "center", "align-items": "center"}
                    ),
                    dcc.RadioItems(
                        id='component-selection',
                        options=[
                            {'label': 'Beam', 'value': 'Beam'},
                            {'label': 'Scatter', 'value': 'Scatter'},
                            {'label': 'All', 'value': 'All'}
                        ],
                        value='All',
                        labelStyle={'display': 'inline-block', 'margin-right': '10px'},
                        style={'width': '100%', 'justify-content': 'center', 'align-items': 'center', 'display': 'flex'}
                    ),
                    dcc.Checklist(id='enable-vertical-slicing', options=[{'label': 'Enable vertical slicing', 'value': 'vertical-slicing'}], value=['vertical-slicing'], style={'width': '100%', 'justify-content': 'center', 'align-items': 'center', 'display': 'flex'})
                ])
            elif tab == 'statistics':
                return html.Div([
                        html.H3("Statistics"),
                        dcc.Loading(
                            html.Div([
                                dcc.Graph(id='statistics-graph', style={'width': '100%', 'height': 'inherit', 'float': 'left'}),
                            ], style={'width': '100%', 'height': '80vh', 'display': 'flex', "justify-content": "center", "align-items": "center"}),
                            id="loading",
                            type="default",
                            style={'width': '100%', 'height': '80vh', 'display': 'flex', "justify-content": "center", "align-items": "center"}
                        ),
                        dcc.Interval(id='progress-interval', interval=500, n_intervals=0, disabled=True),
                        html.Progress(id='stats-progress-bar', value=0, max=100, style={'width': '100%'}),
                    ])

        @self.app.callback(
            [
                Output('statistics-graph', 'figure'),
                Output('stats-progress-bar', 'value'),
                Output('progress-interval', 'disabled')
            ],
            [
                Input('statistics-list', 'value'),
                Input('progress-interval', 'n_intervals')
            ]
        )
        def update_statistics_graph(selected_statistics, n_intervals):
            fig = go.Figure()
            start_calc = selected_statistics and MainWindow.file_parsing_progress == 100 and MainWindow.last_statistics_calculated != selected_statistics
            if start_calc:
                MainWindow.last_statistics_calculated = selected_statistics
                MainWindow.file_parsing_progress = 0
                if 'energy' in selected_statistics:
                    energies = []
                    for i, file in enumerate(self.file_list):
                        metadata: RadiationFieldMetadataHeaderV1 = FieldStore.peek_metadata(file).get_header()
                        energies.append(metadata.simulation.tube.max_energy_eV / 1000.0)  # convert to keV
                        MainWindow.file_parsing_progress = (i + 1) / len(self.file_list) * 100
                    fig.add_trace(go.Histogram(x=energies, name='Energy in keV', nbinsx=int(max(energies))))
                elif 'angles' in selected_statistics:
                    alpha_angles = []
                    beta_angles = []
                    for i, file in enumerate(self.file_list):
                        metadata: RadiationFieldMetadataHeaderV1 = FieldStore.peek_metadata(file).get_header()
                        direction = metadata.simulation.tube.radiation_direction
                        alpha = np.degrees(np.arccos(direction.x))  # Angle with x-axis
                        beta = np.degrees(np.arccos(direction.y))   # Angle with y-axis
                        alpha_angles.append(alpha)
                        beta_angles.append(beta)
                        MainWindow.file_parsing_progress = (i + 1) / len(self.file_list) * 100
                    fig.add_trace(go.Histogram(x=alpha_angles, name='Alpha angle in °'))
                    fig.add_trace(go.Histogram(x=beta_angles, name='Beta angle in °'))
                elif 'source_distance' in selected_statistics:
                    source_distances = []
                    for i, file in enumerate(self.file_list):
                        metadata: RadiationFieldMetadataHeaderV1 = FieldStore.peek_metadata(file).get_header()
                        length = np.linalg.norm(np.array([metadata.simulation.tube.radiation_direction.x, metadata.simulation.tube.radiation_direction.y, metadata.simulation.tube.radiation_direction.z]))
                        source_distances.append(length)
                        MainWindow.file_parsing_progress = (i + 1) / len(self.file_list) * 100
                    fig.add_trace(go.Histogram(x=source_distances, name='Source Distance'))

            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='#2e2e2e',
                plot_bgcolor='#2e2e2e',
                font=dict(color='#ffffff')
            )
            return fig, str(MainWindow.file_parsing_progress), start_calc

        @self.app.callback(
            [
                Output('file-list', 'children'),
                Output('directory-list', 'options'),
                Output('directory-list', 'placeholder')
            ],
            [
                Input('select-directory-btn', 'n_clicks'),
                Input('window-initialized', 'data')
            ]
        )
        def open_dataset(n_clicks, data):
            if n_clicks:
                directory = None
                def on_directory_selected(selected_directory):
                    nonlocal directory
                    directory = selected_directory
                self.window.evaluate_js('pywebview.api.open_directory_dialog()', on_directory_selected)

                while directory is None:
                    pass

                directory = os.path.abspath(directory)
                directory = os.path.normpath(directory)
                self.last_directories.add({'label': os.path.basename(directory), 'value': directory})
                self.dataset_path = directory

            if not data or n_clicks:
                self.file_list = [os.path.normpath(os.path.join(self.dataset_path, f)) for f in os.listdir(self.dataset_path) if f.endswith(".rf3")]
                if len(self.file_list) == 0 and os.path.exists(os.path.join(self.dataset_path, "fields")):
                    self.dataset_path = os.path.join(self.dataset_path, "fields")
                    self.file_list = [os.path.normpath(os.path.join(self.dataset_path, f)) for f in os.listdir(self.dataset_path) if f.endswith(".rf3")]

                if self.dir_watcher is not None:
                    self.dir_watcher.stop()
                self.dir_watcher = Observer()
                self.dir_watcher.schedule(self, self.dataset_path, recursive=False)
                self.dir_watcher.start()
                return [html.Li(os.path.basename(f), id={"type": "file-list-item", "index": i}) for i, f in enumerate(self.file_list)], list(self.last_directories), self.dataset_path
            return [], list(self.last_directories), self.dataset_path

        self.last_prev_clicks = 0
        self.last_next_clicks = 0

        @self.app.callback(
            Output('viewer-graph', 'figure'),
            [
                Input('layer-selection', 'value'),
                Input('component-selection', 'value'),
                Input('vertical-slider', 'value'),
                Input('enable-vertical-slicing', 'value'),
                Input('prev-btn', 'n_clicks'),
                Input('next-btn', 'n_clicks'),
                #Input({'type': 'file-list-item', 'index': dash.dependencies.ALL}, 'n_clicks'),
            ]
        )
        def update_viewer_plot(layer_selection, component_selection, vertical_slider, enable_vertical_slicing, prev_clicks, next_clicks): # layer_selection, component_selection, vertical_slider, , selected_file
            if len(self.file_list) == 0:
                return go.Figure()
            selected_file = self.file_list[self.file_idx]
            if prev_clicks and prev_clicks > self.last_prev_clicks:
                self.on_press_previous()
                self.last_prev_clicks = prev_clicks
                selected_file = self.file_list[self.file_idx]
            if next_clicks and next_clicks > self.last_next_clicks:
                self.on_press_next()
                self.last_next_clicks = next_clicks
                selected_file = self.file_list[self.file_idx]
            if component_selection:
                if component_selection == 'Beam':
                    self.plotter.plot_components = FieldComponent.Beam
                elif component_selection == 'Scatter':
                    self.plotter.plot_components = FieldComponent.Scatter
                else:
                    self.plotter.plot_components = FieldComponent.All
            if layer_selection:
                if layer_selection == 'Kerma':
                    self.plotter.plot_infos = PlotInformation.Doserate
                elif layer_selection == 'Hits':
                    self.plotter.plot_infos = PlotInformation.Hits
                elif layer_selection == 'Directions':
                    self.plotter.plot_infos = PlotInformation.Direction
                elif layer_selection == 'Errors':
                    self.plotter.plot_infos = PlotInformation.Errors
                else:
                    self.plotter.plot_infos = PlotInformation.Energy
            vertical_slice_height = None
            if vertical_slider:
                vertical_slice_height = vertical_slider / 100.0
            if isinstance(enable_vertical_slicing, list):
                if 'vertical-slicing' not in enable_vertical_slicing:
                    vertical_slice_height = None
            
            # Replace with logic to generate Plotly figure based on selected files and other inputs
            fig = self.plotter.plot_field(
                FieldStore.load(selected_file),
                title=os.path.basename(selected_file),
                metadata=FieldStore.load_metadata(selected_file),
                vertical_slice_height=vertical_slice_height
            )
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='#2e2e2e',
                plot_bgcolor='#2e2e2e',
                font=dict(color='#ffffff')
            )
            return fig

        @self.app.callback(
            Output('stats-plot', 'figure'),
            [Input('file-list', 'value')]
        )
        def update_stats_plot(selected_files):
            # Replace with logic to generate Plotly figure based on selected files
            fig = go.Figure()
            # Example plot
            fig.add_trace(go.Bar(x=['A', 'B', 'C'], y=[1, 3, 2], name='Example'))
            fig.update_layout(
                template='plotly_dark',
                paper_bgcolor='#2e2e2e',
                plot_bgcolor='#2e2e2e',
                font=dict(color='#ffffff')
            )
            return fig

        @self.app.callback(
            Output('export-btn', 'n_clicks'),
            [Input('export-btn', 'n_clicks')]
        )
        def export_plot(n_clicks):
            if n_clicks:
                file = 'mocked_path.png'  # Mocked file save dialog
                self.viewer_plot.save_figure(file)
            return 0

    def parse_dataset(self):
        self.enqueue_gui_call(lambda : self.window['-File-Count-'].update(value="Fields Count: " + str(len(self.file_list))))
        stats = RadSimPlotter(infos=self.plotter.plot_infos).parse_dataset(self.dataset_path)
        self.enqueue_gui_call(lambda: self.window['-Energy-Range-'].update(value=f"Energy range: [{'{:1.2f}'.format(stats[:, 0].min())}keV .. {'{:1.2f}'.format(stats[:, 0].max())}keV]"))
        self.enqueue_gui_call(lambda: self.window['-alpha-angle-range-'].update(value=f"Alpha angle range: [{stats[:, 1].min()}° .. {stats[:, 1].max()}°]"))
        self.enqueue_gui_call(lambda: self.window['-beta-angle-range-'].update(value=f"Beta angle range: [{stats[:, 2].min()}° .. {stats[:, 2].max()}°]"))
        self.enqueue_gui_call(lambda: self.stats_plot.draw_figure(RadSimPlotter(infos=self.plotter.plot_infos).plot_hists(stats, ["Energy in keV", "Alpha angle in °", "Beta angle in °"], line_color=sg.theme_button_color()[0])))

    def on_file_removed_from_dataset(self, file: str):
        path = os.path.join(self.dataset_path, file)
        self.file_list.remove(path)
        self.file_list_box.update([os.path.basename(f) for f in self.file_list])
        Thread(target=self.parse_dataset).start()

    def on_new_file_in_dataset(self, file: str):
        path = os.path.join(self.dataset_path, file)
        if path not in self.file_list:
            self.file_list.append(path)
            self.enqueue_gui_call(lambda: self.file_list_box.update([os.path.basename(f) for f in self.file_list]))
            Thread(target=self.parse_dataset).start()
        else:
            if self.file_list.index(path) == self.file_idx:
                Thread(target=self.update_viewer_plot).start()

    def on_press_previous(self):
        self.file_idx -= 1
        if self.file_idx < 0:
            self.file_idx = len(self.file_list) - 1
        Thread(target=self.update_viewer_plot).start()

    def update_viewer_plot(self):
        self.enqueue_gui_call(lambda: self.window["loading"].update(visible=True))
        plotter = RadSimPlotter(show_direction=self.show_radiation_source, infos=self.plotter.plot_infos, norm=self.norm)
        field = plotter.load_file(self.file_list[self.file_idx], self.plotter.plot_infos)
        self.enqueue_gui_call(lambda: self.viewer_plot.draw_figure(plotter.plot_field(field)))
        self.enqueue_gui_call(lambda: self.caption_lbl.update(f"Plot: {os.path.basename(self.file_list[self.file_idx])}"))
        self.enqueue_gui_call(lambda: self.window["loading"].update(visible=False))

    def on_press_next(self):
        self.file_idx = (self.file_idx + 1) % len(self.file_list)
        Thread(target=self.update_viewer_plot).start()

    def __process_gui_thread_callbacks(self):
        count = 0
        while len(self.gui_calls) > 0 and count < 20:
            self.gui_calls.pop()()
            count += 1

    def enqueue_gui_call(self, callback: Callable[[], None]):
        self.gui_calls.append(callback)

    def run(self):
        port = 8050
        #webview.config['use_cef'] = True
        dash_thread = Thread(target=lambda: self.app.run_server(debug=False, port=port), daemon=True)
        dash_thread.start()
        self.window = webview.create_window("Dataset Explorer", f"http://localhost:{port}", width=1600, height=900, resizable=True, js_api=API())
        webview.start()


class API:
    def open_directory_dialog(self):
        root = tk.Tk()
        root.withdraw()
        directory = filedialog.askdirectory()
        root.destroy()
        return directory
