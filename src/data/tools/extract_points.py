'''Helper to Extract points from GEE'''

import ee
import geemap
import ipywidgets as widgets
from ipyleaflet import WidgetControl


def retrieve_points(gee_map):
    """
    Retrieve points interactively from a map using user interaction.

    Parameters:
        gee_map (object): The GEE map object.

    Returns:
        list: List of selected points (latitude, longitude).

    Example:
        points = retrieve_points(gee_map)
    """
    gee_map.points = []

    # Add an output widget to the map
    output_widget = widgets.Output(layout={'border': '1px solid black'})
    output_control = WidgetControl(widget=output_widget, position='bottomright')
    gee_map.add_control(output_control)

    # Capture user interaction with the map
    def handle_interaction(**kwargs):
        latlon = kwargs.get('coordinates')
        if kwargs.get('type') == 'click':
            gee_map.default_style = {'cursor': 'wait'}

            with output_widget:
                output_widget.clear_output()
                print(latlon)
            gee_map.points.append(latlon)
        gee_map.default_style = {'cursor': 'pointer'}

    gee_map.on_interaction(handle_interaction)

    return gee_map.points
