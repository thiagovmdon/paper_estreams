# -*- coding: utf-8 -*-
"""
This file is part of the EStreams catalogue/dataset. See https://github.com/ 
for details.

Coded by: Thiago Nascimento
"""

import geopandas as gpd
from shapely.geometry import Point
from plotly.offline import plot
#import contextily as cx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
import warnings
import datetime
import matplotlib.dates as mdates

import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


warnings.simplefilter(action='ignore', category=Warning)


# This function generate a dataframe with the number of measurements at the daily, monthly and annual time-steps:
# This function assumes that you give as input a time-series at the daily time-step.
# For the daily, the function computes the total number of measurements. For the monthly and yearly it gives both the number of 
# years and month with any, and completed (no gaps within the month or year). Both information might be usefull. 

def count_num_measurements(timeseries: pd.pandas.core.frame.DataFrame):
    """
    Inputs
    ------------------
    timeseries: dataset[Index = Datetime; columns = [Measurements]]: 
        dataframe with datetime as the index, and with each column representing one measurement. 
        It assumes that the gaps in the measurements are stored as np.nan
    Returns
    --------------------
    pandas.DataFrame [n x 6] with columns:
        'Code': Code of the station.
        
        'num_daily': Number of daily measurements.
        
        'num_monthly': Number of of months with any measurement.
        
        'num_yearly': Number of of months with any measurement.
        
        'num_monthly_complete': Number of months with complete measurements.
        
        'num_yearly_complete': Number years with complete measurements.
    """
    # First we create a dataframe for our measurements:
    num_measurements_df = pd.DataFrame(index = timeseries.columns)
    num_measurements_df.index.name = "Code"
    
    # Here we proceed at the daily time-step:
    num_measurements_df["num_daily_obs"] = timeseries.count()
    
    # Now we do the computation for the monthly step:
    timeseries_monthly = timeseries.resample('M').count() # First we count the number of days with non NaN values
    timeseries_monthly.replace(0, np.nan, inplace = True)
    
    num_measurements_df["num_monthly"] = (timeseries_monthly > 0).sum()
    num_measurements_df["num_monthly_complete"] = (timeseries_monthly >= 28).sum()
    
    # Now we do the computation for the yearly step:
    timeseries_yearly = timeseries.resample('Y').count() # First we count the number of days with non NaN values
    timeseries_yearly.replace(0, np.nan, inplace = True)
    
    num_measurements_df["num_yearly"] = (timeseries_yearly > 0).sum()
    num_measurements_df["num_yearly_complete"] = (timeseries_yearly >= 365).sum()
    
    return num_measurements_df


#%% Compute the longest continuous range with no gaps for each gauge (column) in the input DataFrame.
    
from tqdm import tqdm

def longest_continuous_period(timeseries: pd.DataFrame):
    """
    Inputs
    ------------------
    timeseries: dataset[Index = Datetime; columns = [Measurements]]: 
        dataframe with datetime as the index, and with each column representing one measurement. 
        It assumes that the gaps in the measurements are stored as np.nan
    Returns
    --------------------
    pandas.Series with index as column names and values as the longest period with no gap.
    """


    # Calculate the longest continuous range with no gaps for each gauge
    longest_gap_periods = pd.DataFrame(index=timeseries.columns, columns=['longest_gap_period'])

    for col in tqdm(timeseries.columns):
        max_gap = 0
        current_gap = 0
        for value in timeseries[col]:
            if pd.notna(value):
                current_gap += 1
                max_gap = max(max_gap, current_gap)
            else:
                current_gap = 0
        longest_gap_periods.at[col, 'longest_gap_period'] = max_gap
    
    return longest_gap_periods



#%% Plot a map of points in a map (update from ploitpointsmap).
def plotpointsmapnew(ax, plotsome: pd.pandas.core.frame.DataFrame, xcoords="lon", ycoords="lat", 
                      crsproj='epsg:4326', showcodes=False, figsizeproj=(15, 30), 
                      markersize_map=3, colorpoints='black', north_arrow=True, set_map_limits=False,
                      minx=0, miny=0, maxx=1, maxy=1):
    """
    Plots points on a map.

    Parameters:
    ----------
    ax : matplotlib.axes._axes.Axes
        The axes on which to plot the points.

    plotsome : pd.pandas.core.frame.DataFrame
        Dataframe containing geographical data.
        Index should be 'Code', and columns should include 'Longitude-X' and 'Latitude-y'.

    xcoords : str, optional
        Name of the column containing X-coordinates (longitude). Default is "lon".

    ycoords : str, optional
        Name of the column containing Y-coordinates (latitude). Default is "lat".

    crsproj : str, optional
        Coordinate reference system. Default is 'epsg:4326'.

    showcodes : bool, optional
        Whether to show codes from the index. Default is False.

    figsizeproj : tuple, optional
        Figure size in inches. Default is (15, 30).

    markersize_map : int, optional
        Size of the markers on the map. Default is 3.

    colorpoints : str, optional
        Color of the points. Default is 'black'.

    north_arrow : bool, optional
        Whether to include a north arrow. Default is True.

    set_map_limits : bool, optional
        Whether to set custom map limits. Default is False.

    minx : float, optional
        Minimum X-coordinate for map limits. Default is 0.

    miny : float, optional
        Minimum Y-coordinate for map limits. Default is 0.

    maxx : float, optional
        Maximum X-coordinate for map limits. Default is 1.

    maxy : float, optional
        Maximum Y-coordinate for map limits. Default is 1.

    Returns:
    ----------
    plt.plot
        The output is a plt.plot with the points spatially distributed in the area. 
        A background map can also be shown if your coordinate system "crsproj" is set to 'epsg:4326'.
    """
    crs = {'init': crsproj}
    geometry = plotsome.apply(lambda row: Point(row[xcoords], row[ycoords]), axis=1)
    geodata = gpd.GeoDataFrame(plotsome, crs=crs, geometry=geometry)
    geodatacond = geodata

    # Ploting:
    geodatacond.plot(ax=ax, color=colorpoints, markersize=markersize_map, legend=False)

    if showcodes:
        geodatacond["Code"] = geodatacond.index
        geodatacond.plot(column='Code', ax=ax)
        for x, y, label in zip(geodatacond.geometry.x, geodatacond.geometry.y, geodatacond.index):
            ax.annotate(label, xy=(x, y), xytext=(1, 1), textcoords="offset points")
        plt.rcParams.update({'font.size': 12})
    else:
        pass

    if not set_map_limits:
        minx, miny, maxx, maxy = geodatacond.total_bounds
    else:
        pass

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    # Plot the north arrow:
    if north_arrow:
        x, y, arrow_length = 0.025, 0.125, 0.1
        ax.annotate('N', xy=(x, y), xytext=(x, y - arrow_length),
                    arrowprops=dict(facecolor='black', width=5, headwidth=15),
                    ha='center', va='center', fontsize=18,
                    xycoords=ax.transAxes)
    else:
        pass



import geopandas as gpd
from shapely.geometry import Point
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl



def plot_num_measurementsmap_subplot(ax, plotsome: pd.DataFrame, xcoords="lon", ycoords="lat", column_labels="num_yearly_complete",
                                     crsproj='epsg:4326', showcodes=False, markersize_map=3, north_arrow=True, 
                                     set_map_limits=False, minx=0, miny=0, maxx=1, maxy=1, color_categories=None, color_mapping=None,
                                     legend_title=None, legend_labels=None, legend_loc='upper left', show_legend = True, 
                                     legend_outside=True, legend_bbox_to_anchor=(0.5, 1)):  # Add legend_outside and legend_bbox_to_anchor parameters:
    """
    Plot data on a subplot with additional options.

    Parameters:
        ax (matplotlib.axes.Axes): The subplot where the data will be plotted.
        plotsome (pd.DataFrame): The data to be plotted.
        xcoords (str): The name of the column containing x-coordinates.
        ycoords (str): The name of the column containing y-coordinates.
        column_labels (str): The name of the column containing data for coloring.
        crsproj (str): The coordinate reference system (CRS) for the data.
        showcodes (bool): Whether to show data labels.
        markersize_map (int): Size of the markers.
        north_arrow (bool): Whether to include a north arrow.
        set_map_limits (bool): Whether to set specific map limits.
        minx (float): Minimum x-axis limit.
        miny (float): Minimum y-axis limit.
        maxx (float): Maximum x-axis limit.
        maxy (float): Maximum y-axis limit.
        color_categories (list): List of color categories for data bins.
        color_mapping (dict): Mapping of color categories to colors.
        legend_title (str): Title for the legend.
        legend_labels (list): Labels for the legend items.
        legend_loc (str): Location of the legend.
        show_legend (bool): Whether to display the legend.
        legend_outside (bool): Whether to place the legend outside the plot.
        legend_bbox_to_anchor (tuple): Position of the legend (x, y).

    Returns:
        None
    """
    # Prepare the data for plotting
    crs = {'init': crsproj}
    geometry = plotsome.apply(lambda row: Point(row[xcoords], row[ycoords]), axis=1)
    geodata = gpd.GeoDataFrame(plotsome, crs=crs, geometry=geometry)
    geodatacond = geodata

    if color_categories is not None and color_mapping is not None:
        geodatacond['color_category'] = pd.cut(geodatacond[column_labels], bins=[c[0] for c in color_categories] + [np.inf], labels=[f'{c[0]}-{c[1]}' for c in color_categories])
    else:
        raise ValueError("Both color_categories and color_mapping must be provided.")

    # Plotting and legend:
    for category, group in geodatacond.groupby('color_category'):
        #group.plot(ax=ax, color=color_mapping[category], markersize=markersize_map, legend=False, label=category)
        group.plot(ax=ax, marker='o', color=color_mapping[category], markersize=markersize_map, legend=False, label=category, edgecolor='none')
    
    if showcodes == True:
        geodatacond["Code"] = geodatacond.index
        geodatacond.plot(column='Code', ax=ax)
        for x, y, label in zip(geodatacond.geometry.x, geodatacond.geometry.y, geodatacond.index):
            ax.annotate(label, xy=(x, y), xytext=(1, 1), textcoords="offset points")
        plt.rcParams.update({'font.size': 12})

    if set_map_limits == False:
        total_bounds = geodatacond.total_bounds
        minx, miny, maxx, maxy = total_bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

    # Plot the legend
    if legend_labels is None:
        legend_labels = [f'{c[0]}-{c[1]}' for c in color_categories]
        
    if show_legend:
        if legend_outside:
            legend = ax.legend(title=legend_title, labels=legend_labels, loc='upper left', bbox_to_anchor=legend_bbox_to_anchor,
                               bbox_transform=ax.transAxes, frameon=False)  # Use bbox_transform to position the legend
        else:
            legend = ax.legend(title=legend_title, labels=legend_labels, loc=legend_loc, frameon=False)
            
        if legend_outside:
            ax.add_artist(legend)
            
    # Plot the north arrow:
    if north_arrow == True:
        x, y, arrow_length = 0.975, 0.125, 0.1

        ax.annotate('N', xy=(x, y), xytext=(x, y - arrow_length),
                    arrowprops=dict(facecolor='black', width=5, headwidth=15),
                    ha='center', va='center', fontsize=18,
                    xycoords='axes fraction')
  
    # Set font family and size using rcParams
    mpl.rcParams['font.family'] = 'Arial'
    mpl.rcParams['font.size'] = 8  # You can adjust this value as needed


# Function to add circular legend
def add_circular_legend(ax, color_mapping, legend_labels, bbox_to_anchor=(0.01, 0.75)):
    """
    Add a circular legend to a subplot.

    Parameters:
        ax (matplotlib.axes.Axes): The subplot to which the legend will be added.
        color_mapping (dict): A mapping of legend labels to marker colors.
        legend_labels (list): List of legend labels.
        bbox_to_anchor (tuple): The legend's bounding box coordinates.

    Returns:
        None
    """
    handles = [Line2D([0], [0], marker='o', color='none', markerfacecolor=color_mapping[key],
                       markeredgecolor='none', markersize=5) for key in color_mapping]
    legend = ax.legend(handles, legend_labels, loc='upper left', bbox_to_anchor=(1, 1), title=legend_title)
    legend.get_frame().set_linewidth(0)  # Remove legend frame
    legend.get_frame().set_facecolor('none')  # Remove legend background
    legend.set_bbox_to_anchor(bbox_to_anchor)  # Adjust legend position


# Function to create histograms inside subplots
def add_hist(axes, data: pd.DataFrame, axes_loc=[0.05, 0.05, 0.15, 0.175], alpha_hist=0.7,
             num_bins=10, x_ticks=[0, 5], base_xaxis=1, xlim_i=0, xlim_f=5):
    """
    Add a histogram to a subplot.

    Parameters:
        data (pandas.Series): Data for the histogram.
        axes (matplotlib.axes.Axes): The subplot where the histogram will be added.
        axes_loc (list): Location and size of the inset axis.
        alpha_hist (float): Alpha value for histogram transparency.
        num_bins (int): Number of histogram bins.
        x_ticks (list): Specific x-axis tick values.
        base_xaxis (int): Minor locator base for x-axis ticks.
        xlim_i (float): Minimum x-axis limit.
        xlim_f (float): Maximum x-axis limit.

    Returns:
        None
    """
    # Create a histogram inset axis within the subplot
    hist_ax = axes.inset_axes(axes_loc)  # Adjust the values as needed
    # Extract the data for the histogram (replace 'column_name' with the actual column you want to plot)
    hist_data = data.dropna()

    # Plot the histogram within the inset axis
    hist_ax.hist(hist_data, bins=num_bins, color='gray', alpha=alpha_hist)
    hist_ax.set_xlabel('')  # Replace with an appropriate label
    hist_ax.set_ylabel('')  # Replace with an appropriate label

    # Hide the axis spines and ticks for the inset axis
    hist_ax.spines['top'].set_visible(False)
    hist_ax.spines['right'].set_visible(False)
    hist_ax.spines['left'].set_visible(False)
    hist_ax.spines['bottom'].set_visible(True)
    hist_ax.set_facecolor('none')
    hist_ax.set_yticklabels(hist_ax.get_yticks(), rotation=90, fontsize=5)

    # Adjust y-tick label alignment for the right y-axis
    hist_ax.yaxis.tick_right()  # Move the y-tick labels to the right side
    hist_ax.yaxis.set_label_position("right")  # Move the y-axis label to the right side

    # Define the specific y-axis tick values you want to show
    hist_ax.set_xticks(x_ticks)

    # Remove y-axis ticks and labels
    hist_ax.set_yticks([])
    hist_ax.set_yticklabels([])

    hist_ax.xaxis.set_minor_locator(plt.MultipleLocator(base=base_xaxis))  # Adjust the base as needed
    # Set x-axis limits (adjust the values as needed)
    hist_ax.set_xlim(xlim_i, xlim_f)

# Function to add a shapefile plot to a subplot
def add_shapefile(axes, shapefile,  linewidth_shp=0.1,
                  xlim_min=-24, xlim_max=45, ylim_min=35, ylim_max=70):
    """
    Add a shapefile plot to a subplot with specified properties.

    Parameters:
        shapefile (GeoDataFrame): The shapefile data to be plotted.
        axes (matplotlib.axes.Axes): The subplot where the shapefile will be added.
        linewidth_shp (float): Width of shapefile boundaries.
        xlim_min (float): Minimum value for the x-axis limit.
        xlim_max (float): Maximum value for the x-axis limit.
        ylim_min (float): Minimum value for the y-axis limit.
        ylim_max (float): Maximum value for the y-axis limit.

    Returns:
        None
    """
    # Set the background color to white
    axes.set_facecolor('white')
    # Plot the shapefile with white facecolor and black boundaries
    shapefile.plot(ax=axes, facecolor='white', edgecolor='black', linewidth=linewidth_shp)
    axes.set_xlim(xlim_min, xlim_max)
    axes.set_ylim(ylim_min, ylim_max)

# Function to find the first date without gaps
def find_first_non_nan_dates(data):
    """
    Find the first date with a non-NaN value for each column in the DataFrame.

    Parameters:
    - data (pd.DataFrame): Input DataFrame with datetime index.

    Returns:
    - pd.Series: A Series containing the first date with a non-NaN value for each column.

    """
    first_non_nan_dates = data.apply(lambda col: col.first_valid_index())
    return first_non_nan_dates

# Function to find the last date without gaps
def find_last_non_nan_dates(data):
    """
    Find the last date with a non-NaN value for each column in the DataFrame.

    Parameters:
    - data (pd.DataFrame): Input DataFrame with datetime index.

    Returns:
    - pd.Series: A Series containing the last date with a non-NaN value for each column.
    """
    last_non_nan_dates = data.apply(lambda col: col.last_valid_index())
    return last_non_nan_dates








