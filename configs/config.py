import os

from configs.config_utils import ConfigUtils


class Config:

    # System parameters
    system = {
        "frequency": 2.4e9
    }

    # Room parameters
    room = {
        "geometry": "square",
        "length": 3,  # meters
        "width": 3,  # meters
        "origin": "center"  # values: {"center", "corner"}
    }

    # Domain of interest parameters -  scatterers lie in this region
    doi = {
        "geometry": "square",
        "length": 1.5,  # meters
        "width": 1.5,  # meters
        "origin": "center",  # values: {"center", "corner"}
        "forward_grids": 250,
        "inverse_grids": 50
    }

    # Sensor parameters
    sensors = {
        "count": 40,
        "transceivers": True
    }
    sensors["positions"] = ConfigUtils.get_sensor_positions(sensors["count"], room["length"], room["width"], room["origin"])
    sensors["links"] = ConfigUtils.get_sensor_links(sensors["count"], sensors["transceivers"])

    # Generated data parameters
    data = {
        "path": os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"),
        "scatterer_path": os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "scatterer_data"),
        "scatterer_image_path": os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "scatterer_images"),
        "field_data_path": os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "field_data"),
    }
