from RadFiled3D.RadFiled3D import FieldStore, HistogramVoxel, CartesianRadiationField, PolarRadiationField, RadiationFieldMetadataHeaderV1, RadiationFieldMetadata
import plotly.graph_objs as go
import numpy as np
import torch
import os
from enum import Enum
from typing import Union
from logging import getLogger


class PlotInformation(Enum):
    Hits = 0
    Doserate = 1
    Spectra = 3
    Energy = 4
    Errors = 5
    Direction = 6

    def get_keys(self):
        if self == PlotInformation.Hits:
            return ["hits"]
        elif self == PlotInformation.Doserate:
            return ["spectrum", "hits"]
        elif self == PlotInformation.Spectra:
            return ["spectrum"]
        elif self == PlotInformation.Energy:
            return ["energy"]
        elif self == PlotInformation.Errors:
            return ["error"]
        elif self == PlotInformation.Direction:
            return ["direction"]
        else:
            return []


class FieldComponent(Enum):
    Beam = 0
    Scatter = 1
    All = 2

    def get_keys(self):
        if self == FieldComponent.Beam:
            return ["xray_beam"]
        elif self == FieldComponent.Scatter:
            return ["scatter_field"]
        else:
            return ["xray_beam", "scatter_field"]


class Noop:
    def forward(self, data):
        return data
    
    def _get_name(self):
        return "Identity"
    

class MaxNorm:
    def forward(self, data):
        return data / torch.max(data)
    
    def _get_name(self):
        return "MaxNorm"


class ClampingMaxNorm:
    def forward(self, data):
        # filter all data above the standard deviation
        stddev = torch.std(data)
        mean = torch.mean(data)
        # set all data to max that is above the standard deviation
        too_large = data > mean + stddev
        max_allowed_value = torch.max(data[~too_large])
        data = torch.clamp(data, max=max_allowed_value, min=0.0)
        return data / torch.max(data)
    
    def _get_name(self):
        return "ClampingMaxNorm"


class LogNorm:
    def forward(self, data):
        value = torch.log(data + 1)
        return value / torch.max(value)
    
    def _get_name(self):
        return "LogNorm"


class RadSimPlotter:
    def __init__(self, show_direction: bool = False, infos: PlotInformation = PlotInformation.Energy, norm = MaxNorm()) -> None:
        self.show_direction = show_direction
        self.show_primary_radiation_direction = False
        self.plot_infos = infos
        self.norm = norm
        self.current_field: Union[CartesianRadiationField, PolarRadiationField] = None
        self.plot_components = FieldComponent.All

    def parse_dataset(self, path: str) -> np.ndarray:
        metadata = [FieldStore.peek_metadata(f) for f in os.listdir(path) if f.endswith(".rf3")]

        stats = np.zeros((len(metadata), 3), dtype=np.float32)
        for i, m in enumerate(metadata):
            m: RadiationFieldMetadataHeaderV1 = m.get_header() if isinstance(m, RadiationFieldMetadata) else m

            # calc alpha and beta angle of polar rotation from direction vector
            direction = m.simulation.tube.radiation_direction
            alpha_angle = np.arctan2(direction.y, direction.x)
            beta_angle = np.arccos(direction.z)
            alpha_angle = np.degrees(alpha_angle)
            beta_angle = np.degrees(beta_angle)

            stats[i, 0] = m.simulation.tube.max_energy_eV / 1000.0
            stats[i, 1] = alpha_angle
            stats[i, 2] = beta_angle

        return stats

    
    def plot_hists(self, data: Union[np.ndarray, torch.Tensor], data_captions: list, dpi: int = 96, line_color: str = "blue") -> go.Figure:
        fig = go.Figure()

        for i in range(data.shape[1]):
            counts, bins = np.histogram(data[:, i])
            fig.add_trace(go.Histogram(
                x=bins[:-1],
                y=counts,
                name=data_captions[i],
                marker_color=line_color,
                opacity=0.75
            ))

        fig.update_layout(
            title="Histograms",
            xaxis_title="Value",
            yaxis_title="Number of files",
            barmode='overlay'
        )

        return fig
    
    def plot_spectrum(self, xyz: tuple[float, float, float], dpi: int = 96, title: str = "") -> go.Figure:
        if isinstance(self.current_field, PolarRadiationField):
            return None
        
        xyz = [
            xyz[0] + self.current_field.get_field_dimensions().x / 2.0,
            xyz[1] + self.current_field.get_field_dimensions().y / 2.0,
            xyz[2] + self.current_field.get_field_dimensions().z / 2.0 if xyz[2] is not None else self.current_field.get_field_dimensions().z / 2.0
        ]

        xyz = (
            xyz[0],
            xyz[2],
            xyz[1]
        )

        if xyz[0] < 0 or xyz[0] >= self.current_field.get_field_dimensions().x or xyz[1] < 0 or xyz[1] >= self.current_field.get_field_dimensions().y or xyz[2] < 0 or xyz[2] >= self.current_field.get_field_dimensions().z:
            print(f"Coordinates {xyz} are out of bounds")
            return None
        
        component_names: list[str] = self.plot_components.get_keys()
        hist_data: np.ndarray = None
        voxel: HistogramVoxel = None
        for component in component_names:
            if not self.current_field.has_channel(component):
                continue
            if hist_data is None:
                voxel = self.current_field.get_channel(component).get_voxel_by_coord("spectrum", xyz[0], xyz[1], xyz[2])
                hist_data = voxel.get_histogram()
            else:
                hist_data = hist_data + self.current_field.get_channel(component).get_voxel_by_coord("spectrum", xyz[0], xyz[1], xyz[2]).get_histogram()
        hist_data = hist_data / len(component_names)
        hist_data = np.nan_to_num(hist_data, nan=0.0, posinf=1.0, neginf=0.0)
        hist_x = (np.arange(len(hist_data)) * voxel.get_histogram_bin_width() + voxel.get_histogram_bin_width()) / 1000.0

        fig = go.Figure(data=[go.Bar(
            x=hist_x,
            y=hist_data,
            marker_color='blue'  # Change this to match the 'plasma' colorscale
        )])
        
        fig.update_layout(
            title=title,
            xaxis_title='Energy (keV)',
            yaxis_title='Probability Density',
            coloraxis_colorscale='plasma',  # Add this to match the color scheme
            paper_bgcolor='rgba(0,0,0,0)',  # Set paper background to alpha zero
            plot_bgcolor='rgba(0,0,0,0)',    # Set plot background to alpha zero
            autosize=True,                   # Ensure the graph fills available space
            margin=dict(l=0, r=0, t=50, b=0) # Adjust margins for better fit
        )

        return fig

    def get_data(self, field: Union[CartesianRadiationField, PolarRadiationField]) -> np.ndarray:
        layer_names: list[str] = self.plot_infos.get_keys()
        component_names: list[str] = self.plot_components.get_keys()

        overall_data = None
        found_components = 0
        for component in component_names:
            if component not in field.get_channel_names():
                if self.plot_components != FieldComponent.All:
                    raise ValueError(f"Field does not contain component {component}")
                else:
                    continue
            found_components += 1
            layers_data = None
            for layer in layer_names:
                if layer not in field.get_channel(component).get_layers():
                    raise ValueError(f"Field does not contain layer {layer} in component {component}")
                layer_data = field.get_channel(component).get_layer_as_ndarray(layer)
                if layers_data is None:
                    layers_data = layer_data
                else:
                    while len(layers_data.shape) > len(layer_data.shape):
                        layer_data = layer_data[..., np.newaxis]
                    layers_data *= layer_data

            if self.plot_infos != PlotInformation.Direction:
                while len(layers_data.shape) > 3:
                    layers_data = np.trapezoid(layers_data, axis=-1)

            if overall_data is None:
                overall_data = layers_data
            else:
                overall_data += layers_data
        
        if overall_data is None and len(field.get_channels()) == 1:
            getLogger().warning("No data found in field, using first channel")
            overall_data_channel = field.get_channels()[0][1]
            if len(overall_data_channel.get_layers()) == 1:
                overall_data = overall_data_channel.get_layer_as_ndarray(overall_data_channel.get_layers()[0])
                found_components = 1
            else:
                getLogger().error("Field contains multiple layers, but not in the known format")
                raise ValueError("Field contains multiple layers, but not in the known format")

        overall_data /= found_components
        return overall_data

    def load_file(self, file: str, info: PlotInformation) -> CartesianRadiationField:
        file = os.path.normpath(file)
        field: CartesianRadiationField = FieldStore.load(file)
        return field

    def normalize(self, data: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        data = self.norm.forward(data)
        return data
    
    def plot_cartesian_field(self, field: CartesianRadiationField, dpi: int = 96, title: str = "", metadata: RadiationFieldMetadata = None, vertical_slice_height: float = None) -> go.Figure:
        data = torch.from_numpy(self.get_data(field))
        if vertical_slice_height is not None:
            data = data[:, int(data.shape[1] * vertical_slice_height), :].squeeze(1)
        else:
            if self.plot_infos != PlotInformation.Direction:
                data = torch.sum(data, dim=1)
            else:
                data = data[:, data.shape[1] // 2, :].squeeze(1)

        if self.plot_infos != PlotInformation.Direction:
            data = self.normalize(data)
        data = torch.nan_to_num(data, nan=0.0, posinf=1e+304, neginf=-1e+304)
        field_dim = field.get_field_dimensions()
        x = np.linspace(0, field_dim.x, data.shape[0]) - field_dim.x / 2.0
        y = np.linspace(0, field_dim.z, data.shape[1]) - field_dim.z / 2.0
        x, y = np.meshgrid(x, y)

        if self.plot_infos != PlotInformation.Direction:
            fig = go.Figure(data=[go.Surface(z=data.numpy(), x=x, y=y, colorscale='plasma')])
        else:
            fig = go.Figure()
            annotations = []
            voxel_size = field.get_voxel_dimensions()
            voxel_size = np.array([voxel_size.x, voxel_size.y, voxel_size.z], dtype=np.float32)
            voxel_length_2d = np.sqrt(voxel_size[0] ** 2 + voxel_size[1] ** 2)
            voxel_count = field.get_voxel_counts()
            voxel_count = np.array([voxel_count.x, voxel_count.y, voxel_count.z], dtype=np.float32)
            field_dimension = voxel_size * voxel_count
            half_field_dimension = field_dimension / 2.0
            for i in range(data.shape[0]):
                for j in range(data.shape[1]):
                    vector2d = data[i, j]
                    vector2d = np.array([vector2d[0], vector2d[2]], dtype=np.float32)
                    length = np.linalg.norm(vector2d)
                    if length == 0.0:
                        continue
                    vector2d /= length
                    x_offset = i * voxel_size[0] - half_field_dimension[0]
                    y_offset = j * voxel_size[1] - half_field_dimension[1]
                    annotations.append(
                        go.layout.Annotation(
                            x=x_offset,
                            y=y_offset,
                            ax=x_offset + vector2d[0] * voxel_length_2d,
                            ay=y_offset + vector2d[1] * voxel_length_2d,
                            xref="x",
                            yref="y",
                            axref="x",
                            ayref="y",
                            showarrow=True,
                            arrowhead=3,
                            arrowsize=1,
                            arrowwidth=1,
                            arrowcolor="blue"
                        )
                    )
            fig.update_layout(
                annotations=annotations,
                scene=dict(
                    xaxis=dict(
                        title='x in m',
                        range=[-half_field_dimension[0], half_field_dimension[0]]
                    ),
                    yaxis=dict(
                        title='y in m',
                        range=[-half_field_dimension[1], half_field_dimension[1]]
                    )
                )
            )

        particles = "?"
        if metadata is not None:
            particles = metadata.get_header().simulation.primary_particle_count

        fig.update_layout(
            title=f"{title} @ {particles:2e} particles",
            scene=dict(
                xaxis_title='x in m',
                yaxis_title='y in m',
                zaxis_title=f'{self.norm._get_name()}(z)'
            )
        )

        if self.show_primary_radiation_direction:
            voxel_size = field.get_voxel_dimensions()
            voxel_size = np.array([voxel_size.x, voxel_size.y, voxel_size.z], dtype=np.float32)
            voxel_count = field.get_voxel_counts()
            voxel_count = np.array([voxel_count.x, voxel_count.y, voxel_count.z], dtype=np.float32)
            field_dimension = voxel_size * voxel_count

            # plot a red line across the graph along the primary radiation direction
            primary_direction_vector = metadata.get_header().simulation.tube.radiation_direction
            primary_direction_vector = np.array([primary_direction_vector.x, primary_direction_vector.z, primary_direction_vector.y], dtype=np.float32)
            primary_direction_vector /= np.linalg.norm(primary_direction_vector)
            primary_direction_vector *= np.max(field_dimension / 2.0)
            middle_z = (data.max() - data.min()) / 2.0 + data.min()
            fig.add_trace(go.Scatter3d(
                x=[-primary_direction_vector[0], primary_direction_vector[0]],
                y=[-primary_direction_vector[1], primary_direction_vector[1]],
                z=[-primary_direction_vector[2] + middle_z, primary_direction_vector[2] + middle_z],
                mode='lines',
                line=dict(color='red', width=6, dash='dot')
            ))
            # Arrow head
            fig.add_trace(go.Cone(
                x=[primary_direction_vector[0] - primary_direction_vector[0] * 0.125],
                y=[primary_direction_vector[1] - primary_direction_vector[1] * 0.125],
                z=[primary_direction_vector[2] + middle_z - primary_direction_vector[2] * 0.125],
                u=[primary_direction_vector[0]],
                v=[primary_direction_vector[1]],
                w=[primary_direction_vector[2]],
                colorscale=[[0, 'red'], [1, 'red']],
                sizemode="absolute",
                sizeref=0.1,
                showscale=False
            ))

        if self.show_direction and metadata is not None:
            direction = field.get_channel("scatter_field").get_layer_as_ndarray("direction")
            direction = np.array([direction.y, direction.x, direction.z], dtype=np.float32)
            origin = metadata.get_header().simulation.tube.radiation_origin
            origin = np.array([origin.y, origin.x, origin.z + 0.5], dtype=np.float32)
            voxel_dim = field.get_voxel_dimensions()
            steps = field.get_voxel_counts()
            length = (steps.x**2 + steps.y**2)**0.5
            voxel_diameter = (voxel_dim.x**2 + voxel_dim.y**2)**0.5
            origin = np.tile(origin, (int(length), 1))
            direction = np.tile(direction, (int(length), 1))
            trace = origin + np.linspace(0, length * voxel_diameter, int(length)).reshape(int(length), 1) * direction

            fig.add_trace(go.Scatter3d(
                x=trace[:, 0], y=trace[:, 1], z=trace[:, 2],
                mode='lines+markers',
                marker=dict(size=2, color='red'),
                line=dict(color='blue', dash='dot')
            ))
        return fig
    
    def plot_polar_field(self, field: PolarRadiationField, dpi: int = 96, title: str = "", metadata: RadiationFieldMetadata = None) -> go.Figure:
        data = torch.from_numpy(self.get_data(field))
        if len(data.shape) > 2:
            data = torch.sum(data, dim=-1)
        data = self.normalize(data)
        data = torch.nan_to_num(data, nan=0.0, posinf=1e+304, neginf=-1e+304)

        assert len(data.shape) == 2, "Data must be 2D"
        
        theta = np.linspace(0, 2 * np.pi, data.shape[1])
        phi = np.linspace(0, np.pi, data.shape[0])
        theta, phi = np.meshgrid(theta, phi)

        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        
        fig = go.Figure(data=[go.Surface(x=x, y=y, z=z, surfacecolor=data.numpy(), colorscale='plasma')])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z'
            )
        )

        return fig

    def plot_field(self, field: Union[CartesianRadiationField, PolarRadiationField], dpi: int = 96, title: str = "", metadata: RadiationFieldMetadata = None, vertical_slice_height: float = None) -> go.Figure:
        #conv3d = nn.Conv3d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1)
        #conv3d.weight.data.fill_(1.0)
        #conv3d.bias.data.fill_(0.0)
        #data = data.unsqueeze(0).unsqueeze(0)
        #data = conv3d(data)
        #data = data.squeeze(0).squeeze(0).detach()

        self.current_field = field

        type_name = field.get_typename()
        if type_name == "CartesianRadiationField":
            return self.plot_cartesian_field(field, dpi, title, metadata, vertical_slice_height)
        elif type_name == "PolarRadiationField":
            return self.plot_polar_field(field, dpi, title, metadata)
        else:
            raise ValueError(f"Unsupported field type {type_name}")
