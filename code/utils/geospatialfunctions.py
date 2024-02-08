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


warnings.simplefilter(action='ignore', category=Warning)


#%% 1. Define a function for plot:

# This function is used for generating a quick map plot with the desired points 
# and a background map in background.

def plotpointsmap(plotsome: pd.pandas.core.frame.DataFrame, crsproj = 'epsg:4326', backmapproj = True,
               showcodes = False, figsizeproj = (15, 30), markersize_map = 3, colorpoints = 'black', north_arrow = True, set_map_limits = False,
                 minx = 0, miny = 0, maxx = 1, maxy = 1):
    """
    Inputs
    ------------------
    plotsome: dataset[Index = Code; columns = [Longitude, Latitude]]: 
        dataframe with the codes as the index, and with at least two columns in the order of
        "Longitude-X" and "Latitude-y" (in EPSG: 4326 as the first ad second columns. 
    
    showcodes: 
        By default it is set as "False". If "True" it will show the codes from the index. 
    
    Returns
    --------------------
    plt.plot: The output is a plt.plot with the points spatially distributed in the area. 
    A background map can be also shown if your coordinate system "crsproj" is set to 'epsg:4326'.
        
    """    
    if backmapproj == True:
        
        crs={'init':crsproj}
        geometry=[Point(xy) for xy in zip(plotsome.iloc[:,0], plotsome.iloc[:,1])]
        geodata=gpd.GeoDataFrame(plotsome,crs=crs, geometry=geometry)
        geodatacond = geodata

        # The conversiojn is needed due to the projection of the basemap:
        geodatacond = geodatacond.to_crs(epsg=3857)

        # Plot the figure and set size:
        fig, ax = plt.subplots(figsize = figsizeproj)

        #Ploting:
        #geodatacond.plot(ax=ax, column='PercentageGaps', legend=True, cax = cax, cmap = "Reds")
        geodatacond.plot(ax=ax, color = colorpoints, markersize = markersize_map, legend = False)
    
        if showcodes == True:
            geodatacond["Code"] = geodatacond.index
            geodatacond.plot(column = 'Code',ax=ax);
            for x, y, label in zip(geodatacond.geometry.x, geodatacond.geometry.y, geodatacond.index):
                ax.annotate(label, xy=(x, y), xytext=(1, 1), textcoords="offset points")
            plt.rcParams.update({'font.size': 12})
    
        else:
            pass
    
        cx.add_basemap(ax)
    
    else:
        
        crs={'init':crsproj}
        geometry=[Point(xy) for xy in zip(plotsome.iloc[:,0], plotsome.iloc[:,1])]
        geodata=gpd.GeoDataFrame(plotsome,crs=crs, geometry=geometry)
        geodatacond = geodata

        # Plot the figure and set size:
        fig, ax = plt.subplots(figsize = figsizeproj)

        #Ploting:
        
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        world.plot(ax = ax, color='white', edgecolor='black')
        
        #geodatacond.plot(ax=ax, column='PercentageGaps', legend=True, cax = cax, cmap = "Reds")
        geodatacond.plot(ax=ax, color = colorpoints, markersize = markersize_map, legend = False)
    
        if showcodes == True:
            geodatacond["Code"] = geodatacond.index
            geodatacond.plot(column = 'Code',ax=ax);
            for x, y, label in zip(geodatacond.geometry.x, geodatacond.geometry.y, geodatacond.index):
                ax.annotate(label, xy=(x, y), xytext=(1, 1), textcoords="offset points")
            plt.rcParams.update({'font.size': 12})
    
        else:
            pass
        
        if set_map_limits == False:
            minx, miny, maxx, maxy = geodatacond.total_bounds
        
        else:
            pass
        
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
    
    # Plot the north arrow:
    if north_arrow == True:
        x, y, arrow_length = 0.025, 0.125, 0.1
        
        ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
        arrowprops=dict(facecolor='black', width=5, headwidth=15),
        ha='center', va='center', fontsize=18,
        xycoords=ax.transAxes)
    else:
        pass
   
    return fig, ax
    
#%% #### 2. Define a function for plot several time-series in a single plot:

# This function is used for generating a quickly several subplots of different
# time series in a single plot. 

def plottimeseries(numr, numc, datatoplot: pd.pandas.core.frame.DataFrame, setylim = False, ymin = 0, ymax = 1, figsizeproj = (18, 11),
                   colorgraph = "blue", linewidthproj = 0.5, linestyleproj = "-",  ylabelplot = "P (mm)",
                   datestart = datetime.date(1981, 6, 1), dateend = datetime.date(2021, 12, 31),
                   setnumberofintervals = False, numberintervals = 2, fontsize_plot = 8):
    
    """
    Inputs
    ------------------
    numr = Number of rows of your figure;
    numc = Numer of columns of your figure;
    datatoplot: dataframe[Index = Datetime; columns = [rain-gauges]]: 

    setylim = It is used when one needs to set a common y-lim for the graphs;
    ymin and ymax = only used when "setylim" is "True";
    figsizeproj = size of the generated figure in inches;
    colorgraph = linecolor of the graphs;
    linewidthproj = linewidth of the graphs;
    linestyleproj = linestyle of the graphs;
    ylabelplot = label of the time-series (assuming all time series are in the same units and type);
    
    datestart and dateend = datetime variable defining the time-interval of the plots;
    setnumberofintervals = It is used when one needs to set manually the number of intervals of 
    the x-axis in years;
    numberintervals = By default it is set to 2-years.
    
    Returns
    --------------------
    plt.plot: The output is a plt.plot with the graphs plot in subplots. 
        
    """   
    
    fig, axs = plt.subplots(int(numr),int(numc), figsize = figsizeproj)

    i = 0

    for col in datatoplot.columns:
    
        plot_data = datatoplot.loc[:,col].dropna()
    
        name = col
    
        num_rows = axs.shape[0]
        num_cols = axs.shape[1]

    
        colauxs = [i for i in range(num_cols) for _ in range(num_rows)] 
        rowauxs = [*range(num_rows)] * num_cols
    
        colaux, rowaux = colauxs[i], rowauxs[i]
    
        axs[rowaux,colaux].plot(plot_data.index.values, plot_data.values, linestyle = linestyleproj, label=col, linewidth = linewidthproj, markersize=2, color = colorgraph)
        axs[rowaux,colaux].set_title(name, loc='left')
        axs[rowaux,colaux].set_xlim([datestart, dateend])
        
        if setnumberofintervals == True:
            axs[rowaux,colaux].xaxis.set_major_locator(mdates.YearLocator(int(numberintervals)))
        
        if setylim == True:
            axs[rowaux,colaux].set_ylim(ymin, ymax)
        else:
            pass
        
        if colaux == 0:
            axs[rowaux,colaux].set_ylabel(ylabelplot)
    
        i = i + 1

    plt.rcParams.update({'font.size': fontsize_plot})
    plt.tight_layout()
    
    return fig, axs
    
#%% 3. Define a function for plot several box-plots in a single plot:

# This function is useful for the plot of several boxplots from a big time-series
# Initiall this function is used in a dataframe of 1898 rain gauges being each labeled with an unique 
# index (Code) and categorized per Federation State (or Cluster). Moreover, the dataframe has a column of
# maximum precipitation per code, which will be used for the boxplots.
# Therefore, the boxplots will be plot per State (and not per Code). 
# For different analysis the code might as well need to be adapted.


def plotboxplots(numr, numc, datatoplot: pd.pandas.core.frame.DataFrame, setylim = False, 
                 ymin = 0, ymax = 1, figsizeproj = (18, 11), ylabelplot = "P (mm)", 
                 Cluster = "Cluster",Descriptor = "Descriptor", 
                 font_size_plot = {'font.size': 12}):
    
    """
    Inputs
    ------------------
    numr = Number of rows of your figure;
    numc = Numer of columns of your figure;
    datatoplot: dataframe[Index = Codes; columns = [Cluster, Statistical descriptor]]: 

    setylim = It is used when one needs to set a common y-lim for the graphs;
    ymin and ymax = only used when "setylim" is "True";
    figsizeproj = size of the generated figure in inches;
    ylabelplot = label of the time-series (assuming all time series are in the same units and type);
    
    Returns
    --------------------
    plt.plot: Boxplot. 
        
    """   
    fig, axs = plt.subplots(int(numr),int(numc), figsize = figsizeproj)

    i = 0
    
    for col in datatoplot[Cluster].unique():
    
        plot_data = datatoplot[datatoplot[Cluster] == col].loc[:,Descriptor]
    
        name = col
        
        num_rows = axs.shape[0]
        num_cols = axs.shape[1]

        
        colauxs = [i for i in range(num_cols) for _ in range(num_rows)] 
        rowauxs = [*range(num_rows)] * num_cols
    
        colaux, rowaux = colauxs[i], rowauxs[i]
        
        # Here we can plot some text in our boxplot plots: 
        text_to_plot = "Number: " + str(len(plot_data))
        axs[rowaux,colaux].boxplot(plot_data)
        axs[rowaux,colaux].set_title(name, loc='left')
        axs[rowaux,colaux].text(0.25, 0.90, text_to_plot, horizontalalignment='center', 
                                verticalalignment='center', transform= axs[rowaux,colaux].transAxes)
        
        if setylim == True:
            axs[rowaux,colaux].set_ylim(ymin, ymax)
        else:
            pass
        
        if colaux == 0:
            axs[rowaux,colaux].set_ylabel(ylabelplot)
        
        
        i = i + 1

    plt.rcParams.update(font_size_plot)
    plt.tight_layout()

    return fig, axs

#%% 4. Make a df.describe considering a cluster:

# This function is useful for the quick computation of the main statistical descriptors
# such as: min, max, median and percentils of an initial time-series per cluster. 
# For example, one may have a initial time-series of several rain-gauge considering 
# monthly precipitation data and information about potential clusters (or regions). then:
    # 1. This function compute the maximum, minimum, average or other descriptor for each rain-gauge;
    # 2. The statistical descriptors of this descriptor are computed per cluster (region). 

def describeclusters(dataset: pd.pandas.core.frame.DataFrame, clusters: pd.pandas.core.frame.DataFrame, 
                     statisticaldescriptor = "mean", clustercolumnname = "Cluster"): 
    
    """
    Inputs
    ------------------

    dataset: dataframe[Index = Datetime; columns = [rain-gauges]]
    clusters: dataframe[Index = Code just as the columns of dataset; columns = clusters: 
    statisticaldescriptor: {"mean", "count", "std", "min", "25%", "50%", "75%", "max"}                    
    clustercolumn: Column cluster's name in the cluster dataframe.
        
    # It is essential that the columns of the dataframe dataset are the same as the index in the dataframe clusters. 
    
    Returns
    --------------------
    stationsdescriptor: dataframe[Index = Rain-gauges; columns = [Clusters, statisticaldescriptor]]
    clustersdescribe: dataframe[Index = Clusters; columns = ["mean", "min", "P25", "950", "25%", "P75",
                                                             "P90", "P95", "P99", "max", "P25 + 1.5IQR"]] 
        
    """   
    fsummary = dataset.describe()
    stationsdescriptor = pd.DataFrame(index = clusters.index, columns= ["Cluster"], data = clusters.loc[:, clustercolumnname].values)
    stationsdescriptor[statisticaldescriptor] = fsummary.T[statisticaldescriptor].values
    
    clustersdescribe = stationsdescriptor.groupby(by=["Cluster"]).mean()
  
    clustersdescribe.rename(columns = {statisticaldescriptor:'mean'}, inplace = True)
    clustersdescribe["min"] = stationsdescriptor.groupby(by=["Cluster"]).min()
    clustersdescribe["P25"] = stationsdescriptor.groupby(by=["Cluster"]).quantile(q = 0.25)
    clustersdescribe["P50"] = stationsdescriptor.groupby(by=["Cluster"]).quantile(q = 0.5)
    clustersdescribe["P75"] = stationsdescriptor.groupby(by=["Cluster"]).quantile(q = 0.75)
    clustersdescribe["P90"] = stationsdescriptor.groupby(by=["Cluster"]).quantile(q = 0.90)
    clustersdescribe["P95"] = stationsdescriptor.groupby(by=["Cluster"]).quantile(q = 0.95)
    clustersdescribe["P99"] = stationsdescriptor.groupby(by=["Cluster"]).quantile(q = 0.99)
    clustersdescribe["max"] = stationsdescriptor.groupby(by=["Cluster"]).max()
    clustersdescribe["Q1+1.5IQR"] = clustersdescribe["P25"] + (clustersdescribe["P75"] - clustersdescribe["P25"])*1.5
    
    return stationsdescriptor, clustersdescribe

#%% 5. Generate a grid centroids table from initial Lat/Lon and spacing data:

# The main application is for generating centroids of sattelite precipitation grid data (e.g., TRMM).
# Observations:

# (a) The latitude and longitude are computed as being from left to right and from upper to down;
# (b) Pay attention on where you have the space positive or negative, for instance, for Paraiba and TRMM, 
# 0.25 is negative for latitude and positive for longitude. Try to pay attention on where is the (0, 0) of 
# the equator and Greenwich. 
# Lat_final and lon_final must be set with one extra, because python does not consider the last.

def generategridcentroids(lat_initial, lat_final, lon_initial, lon_final, lat_spacing, lon_spacing, crsproj = 'epsg:4326'):
    """
    Inputs
    ------------------

    lat_initial: Latitude value in the centroid located at the upper left (origin)
    lat_final: Latitude value in the centroid located at the lower right + lat_spacing *
    lon_initial: Longitude value in origin (upper left) 
    lon_final: Longitude value in the centroid located at the lower right + lon_spacing *
    lat_spacing: latitude resolution
    lon_spacing: longitude reolution
    crsproj: If you are using WGS84, you the code you plot for you a background map, if not, it will just plot the points.
    
    * Lat_final and lon_final must be set with one extra, because python does not consider the last.
    
    Returns
    --------------------
    coord_grids: dataframe[Index = Centroid IDs; columns = [Lat, Lon]]
    plt.plot: Scatter plot showing the generated grid for conference. 
        
    """       
    
    
    
    lat = np.arange(lat_initial, lat_final, lat_spacing)
    lon = np.arange(lon_initial, lon_final, lon_spacing)
    
    num_rows = len(lat) * len(lon)
    coord_grids = pd.DataFrame(np.nan, index = range(num_rows), columns = [['Lat', 'Lon']])
    
    z = 0
    for i in lat:
        j = 0
        for j in lon:
            coord_grids.iloc[z, 0]  = i
            coord_grids.iloc[z, 1] = j
            z = z + 1
    coord_grids = pd.DataFrame(coord_grids)
    
    if crsproj == 'epsg:4326':
        coords_df = pd.DataFrame({'GridID': range(len(coord_grids)),
                         'Lat': coord_grids.Lat.values[:,0],
                         'Lon': coord_grids.Lon.values[:,0]})
        
        crs = {'init': crsproj}
        
        geometry=[Point(xy) for xy in zip(coords_df["Lon"], coords_df["Lat"])]
        geodata=gpd.GeoDataFrame(coords_df,crs=crs, geometry=geometry)
        geodatacond = geodata
        # The conversiojn is needed due to the projection of the basemap:
        geodatacond = geodatacond.to_crs(epsg=3857)

        # Plot the figure and set size:
        fig, ax = plt.subplots()

        #Organizing the legend:
        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="5%", pad=0.1)

        #Ploting:
        geodatacond.plot(ax=ax)
        cx.add_basemap(ax)
    else:
        plt.scatter(coord_grids.Lon, coord_grids.Lat)
        
    return coord_grids

#%% 6. Generate a summary with some information about data gaps:
def summarygaps(df: pd.pandas.core.frame.DataFrame, coordsdf: pd.pandas.core.frame.DataFrame):
    """
    Inputs
    ------------------
    df: dataset[n x y]: 
        dataframe with already set an datetime index, and unique codes as columns. 
    
    coordsdf: dataset[y x 2]: 
        dataframe with its index as the same codes as the df columns, plus a X and Y gepgraphic coordinates (please follow this order). 
    
    Returns
    --------------------
    pandas.DataFrame [n x 4] with columns:
        'CoordX': Coordinates X
        
        'CoordY': Coordinates Y
        
        'NumGaps': Number of gaps per column
        
        'PercentageGaps': Percentage of gaps per column (%)
    """
    
    
    # Dealing with the data dataframe:
    df.index.name = 'dates'
    
    # Dealing with the coords dataframe:   
    coordsdf.index.name = 'Code'
    
    numrows= df.shape[0] #Total time lenght 

    # Calculate the percentage of failures per point:
    desc = df.describe()
    percerrorsdf = pd.DataFrame(index = coordsdf.index)
    
    percerrorsdf["CoordX"] = coordsdf.iloc[:,0]
    percerrorsdf["CoordY"] = coordsdf.iloc[:,1]
    
    percerrorsdf["NumGaps"] = numrows - desc.iloc[0,:]
    percerrorsdf["PercentageGaps"] = (1 - desc.iloc[0,:]/numrows)*100
    
    return percerrorsdf

#%% 7. Plot the data gaps spatially (this function receives as input the output from function (6)):
def plotgaps(summarygapsstations: pd.pandas.core.frame.DataFrame, crsproj = 'epsg:4326', 
             backmapproj = True, figsizeproj = (15, 30), cmapproj = "Reds",
             legend_title = "Percentage of gaps (%)", legend_orientation = "vertical"):
    """
    Inputs
    ------------------
    summarygapsstations: dataset[y x 4]: 
        The same dataframe output from the fillinggaps.summarygaps function.
    
    Returns
    --------------------
    plt.plot: The output is a plt.plot with the points spatially distributed in the area, and with a legend bar 
        showing the percentage of gaps (from 1 to 100). A background map can be also shown if your coordinate system 
        "crsproj" is set to 'epsg:4326'.
        
    """
    if backmapproj == True:
        
        
        crs = {'init': crsproj}
        geometry = [Point(xy) for xy in zip(summarygapsstations["CoordX"], summarygapsstations["CoordY"])]
        geodata=gpd.GeoDataFrame(summarygapsstations,crs=crs, geometry=geometry)
        geodatacond = geodata

        # The conversiojn is needed due to the projection of the basemap:
        geodatacond = geodatacond.to_crs(epsg=3857)

        # Plot the figure and set size:
        fig, ax = plt.subplots(figsize = figsizeproj)

        #Organizing the legend:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)

        #Ploting:
        geodatacond.plot(ax=ax, column='PercentageGaps', legend=True, cax = cax, cmap = "Reds", 
                         legend_kwds={'label': legend_title,
                        'orientation': legend_orientation})
        cx.add_basemap(ax)
        
        
    else:
        crs = {'init': crsproj}
        geometry = [Point(xy) for xy in zip(summarygapsstations["CoordX"], summarygapsstations["CoordY"])]
        geodata=gpd.GeoDataFrame(summarygapsstations,crs=crs, geometry=geometry)
        geodatacond = geodata

        # Plot the figure and set size:
        fig, ax = plt.subplots(figsize = figsizeproj)

        #Organizing the legend:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        
        #Ploting:
        geodatacond.plot(ax=ax, column='PercentageGaps', legend=True, cax = cax, cmap = cmapproj, 
                         legend_kwds={'label': legend_title,
                        'orientation': legend_orientation})
            
    
    return fig, ax
#%% 8. Plot the data gaps spatially - version 2 (this function receives as input the output from function (6)):

def plotgapsmap(summarygapsstations: pd.pandas.core.frame.DataFrame, crsproj = 'epsg:4326', 
             backmapproj = True, figsizeproj = (20, 100), 
             cmapproj = "Reds", pad_map = -0.01, markersize_map = 5, linewidth_marker = 0.1,
             legend_title = "Percentage of gaps (%)", legend_orientation = "vertical",
             set_map_limits = False, minx = 0, miny = 0, maxx = 1, maxy = 1,
             north_arrow = True):
    
    """
    Inputs
    ------------------
    summarygapsstations: dataset[y x 4]: 
        The same dataframe output from the fillinggaps.summarygaps function.
    
    Returns
    --------------------
    plt.plot: The output is a plt.plot with the points spatially distributed in the area, and with a legend bar 
        showing the percentage of gaps (from 1 to 100). A background map can be also shown if your coordinate system 
        "crsproj" is set to 'epsg:4326'.
        
    """
    if backmapproj == True:
        
        
        crs = {'init': crsproj}
        geometry = [Point(xy) for xy in zip(summarygapsstations["CoordX"], summarygapsstations["CoordY"])]
        geodata=gpd.GeoDataFrame(summarygapsstations,crs=crs, geometry=geometry)
        geodatacond = geodata

        # The conversiojn is needed due to the projection of the basemap:
        geodatacond = geodatacond.to_crs(epsg=3857)

        # Plot the figure and set size:
        fig, ax = plt.subplots(figsize = figsizeproj)

        #Organizing the legend:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="1%", pad = pad_map)

        #Ploting:
        geodatacond.plot(ax=ax, column='PercentageGaps', legend=True, cax = cax, cmap = "Reds",
                         vmin=0, vmax=100, 
                         legend_kwds={'label': legend_title,
                        'orientation': legend_orientation}, markersize = markersize_map,
                        edgecolor="black", linewidth= linewidth_marker)
        cx.add_basemap(ax)
        
        if set_map_limits == False:
            pass
        
        else:
            ax.set_xlim(minx, maxx)
            ax.set_ylim(miny, maxy)  
        
    else:
        crs = {'init': crsproj}
        geometry = [Point(xy) for xy in zip(summarygapsstations["CoordX"], summarygapsstations["CoordY"])]
        geodata=gpd.GeoDataFrame(summarygapsstations,crs=crs, geometry=geometry)
        geodatacond = geodata

        # Plot the figure and set size:
        fig, ax = plt.subplots(figsize = figsizeproj)

        #Organizing the legend:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="1%", pad= pad_map)
        
        #Ploting:
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        world.plot(ax = ax, color='white', edgecolor='black')
        
        geodatacond.plot(ax=ax, column='PercentageGaps', legend=True, cax = cax, cmap = cmapproj,
                         vmin=0, vmax=100,
                         legend_kwds={'label': legend_title,
                        'orientation': legend_orientation},
                         markersize = markersize_map,
                         edgecolor="black", linewidth= linewidth_marker)
        
        if set_map_limits == False:
            minx, miny, maxx, maxy = geodatacond.total_bounds
        
        else:
            pass
        
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
    
    # Plot the north arrow:
    if north_arrow == True:
        x, y, arrow_length = 0.025, 0.125, 0.1
        
        ax.annotate('N', xy=(x, y), xytext=(x, y-arrow_length),
        arrowprops=dict(facecolor='black', width=5, headwidth=15),
        ha='center', va='center', fontsize=18,
        xycoords=ax.transAxes)
    else:
        pass  

        # This part is for the scale bar:
        #from matplotlib_scalebar.scalebar import ScaleBar

        #from shapely.geometry.point import Point as Point2

        #points = gpd.GeoSeries([Point(-74.20, 40.5), Point(-74.5, 40.5)], crs=4326)  # Geographic WGS 84 - degrees
        #points = points.to_crs(32619) # Projected WGS 84 - meters
        
        #distance_meters = points[0].distance(points[1])

        #ax.add_artist(ScaleBar(distance_meters, dimension="si-length", units="km"))
        
   
    
    # This part can be added if you want to show a legend for the proportion in the circle sizes:
    #import matplotlib.lines as mlines
    
    # some bins to indicate size in legend
    #bins_aux = [0, 1000, 10000, 100000]
    #labels_aux = ["Catchment area (km2)", "0 - 1,000", "1,000 - 10,000", ">10,000" ]
    #_, bins = pd.cut(markersize_map, bins=bins_aux, precision=0, retbins=True)
    ## create second legend
    #ax.add_artist(
    #    ax.legend(
    #        handles=[
    #            mlines.Line2D(
    #                [],
    #                [],
    #                color="white",
    #                lw=0,
    #                marker="o",
    #                markersize = np.sqrt(b/100),
    #                markeredgewidth=0,
    #                label = labels_aux[i],
    #            )
    #            for i, b in enumerate(bins)
    #        ],
    #        #loc=4,
    #    fontsize=12, labelspacing = 2, frameon = False)
    #)
        
    return fig, ax

#%% 9. Plot the Gannt chart of our time-series:
# If you are trying to plot more than 50 points at once maybe the visualization will not be the best. 

def plotganntchart(timeseriesfinal_gantt: pd.pandas.core.frame.DataFrame, figsize_chart = (40, 20), 
                   showcodes = False,
                   color_chart = "blue", fontsize_chart = 12, facecolor_chart = "white", 
                   title_chart = "Title"):
    
    """
    Inputs
    ------------------
    summarygapsstations: dataset[y x 4]: 
        The same dataframe output from the fillinggaps.summarygaps function.
    
    Returns
    --------------------
    plt.plot: The output is a plt.plot with the points spatially distributed in the area, and with a legend bar 
        showing the percentage of gaps (from 1 to 100). A background map can be also shown if your coordinate system 
        "crsproj" is set to 'epsg:4326'.
        
    """
    new_rows = [timeseriesfinal_gantt[s].where(timeseriesfinal_gantt[s].isna(), i) for i, s in enumerate(timeseriesfinal_gantt, 1)]
    # To increase spacing between lines add a number to i, eg. below:
    # [df[s].where(df[s].isna(), i+3) for i, s in enumerate(df, 1)]
    new_df = pd.DataFrame(new_rows)

    ### Plotting ###

    fig, ax = plt.subplots() # Create axes object to pass to pandas df.plot()
    ax = new_df.transpose().plot(figsize = figsize_chart, ax = ax, legend=False, fontsize = fontsize_chart, color = color_chart)
    list_of_sites = new_df.transpose().columns.to_list() # For y tick labels
    x_tick_location = np.arange(1.0, len(new_df) + 1, 1.0) # For y tick positions
    ax.set_yticks(x_tick_location) # Place ticks in correct positions
    
    ax.set_title(title_chart)
    ax.set_facecolor(facecolor_chart)
    
    ax.set_yticklabels("")
    
    if showcodes == True:
        ax.set_yticklabels(list_of_sites) # Update labels to site names
    
    
    return fig, ax


#%% 10. Define a function for plot several Gannt-plots in a single plot:

# This function is useful for the plot of several boxplots from a big time-series
# Initiall this function is used in a dataframe of 1898 rain gauges being each labeled with an unique 
# index (Code) and categorized per Federation State (or Cluster). Moreover, the dataframe has a column of
# maximum precipitation per code, which will be used for the boxplots.
# Therefore, the boxplots will be plot per State (and not per Code). 
# For different analysis the code might as well need to be adapted.


def plotganntplots(numr, numc, timeseriesfinal_used: pd.pandas.core.frame.DataFrame, summarygapsstations:pd.pandas.core.frame.DataFrame,
                   setylim = False,  ymin = 0, ymax = 100, figsize_chart = (40, 20), ylabelplot = "P (mm)", 
                   Cluster = "Cluster", Descriptor = "Descriptor", time_range_year_ini = "1981", time_range_year_fin = "2021", 
                   color_chart = "blue", fontsize_chart = 12, facecolor_chart = "white", title_chart = "Title"):
    
    """
    Inputs
    ------------------
    numr = Number of rows of your figure;
    numc = Numer of columns of your figure;
    datatoplot: dataframe[Index = Codes; columns = [Cluster, Statistical descriptor]]: 

    setylim = It is used when one needs to set a common y-lim for the graphs;
    ymin and ymax = only used when "setylim" is "True";
    figsizeproj = size of the generated figure in inches;
    ylabelplot = label of the time-series (assuming all time series are in the same units and type);
    
    Returns
    --------------------
    plt.plot: Boxplot. 
        
    """
    countries = summarygapsstations.Country.unique().tolist() 
        
    fig, axs = plt.subplots(int(numr),int(numc), figsize = figsize_chart)

    i = 0
    
    for country in countries:
        
        name = country
        title_chart = country
        
        num_rows = axs.shape[0]
        num_cols = axs.shape[1]
        
        colauxs = [i for i in range(num_cols) for _ in range(num_rows)] 
        rowauxs = [*range(num_rows)] * num_cols
    
        colaux, rowaux = colauxs[i], rowauxs[i]
        
        
        idcondition = summarygapsstations[summarygapsstations.Country == country].index.tolist()
        timeseriesfinal_gantt = timeseriesfinal_used.loc[time_range_year_ini:time_range_year_fin, idcondition]
    
        new_rows = [timeseriesfinal_gantt[s].where(timeseriesfinal_gantt[s].isna(), i) for i, s in enumerate(timeseriesfinal_gantt, 1)]
        # To increase spacing between lines add a number to i, eg. below:
        # [df[s].where(df[s].isna(), i+3) for i, s in enumerate(df, 1)]
        new_df = pd.DataFrame(new_rows)

        ### Plotting ###
        ax = new_df.transpose().plot(figsize = figsize_chart, ax = axs[rowaux,colaux], legend=False, fontsize = fontsize_chart, 
                                     color = color_chart)
        list_of_sites = new_df.transpose().columns.to_list() # For y tick labels
        x_tick_location = np.arange(1.0, len(new_df) + 1, 1.0) # For y tick positions
        axs[rowaux,colaux].set_yticks(x_tick_location) # Place ticks in correct positions
        axs[rowaux,colaux].set_yticklabels("") # Update labels to site names
        axs[rowaux,colaux].set_title(title_chart)
        axs[rowaux,colaux].set_facecolor(facecolor_chart)
    
        i = i + 1

    plt.rcParams.update({'font.size': 12})
    plt.tight_layout()

    return fig, axs



#%% 11. Plot histograms:

# This function is useful for the plot of several boxplots from a big time-series
# Initiall this function is used in a dataframe of 1898 rain gauges being each labeled with an unique 
# index (Code) and categorized per Federation State (or Cluster). Moreover, the dataframe has a column of
# maximum precipitation per code, which will be used for the boxplots.
# Therefore, the boxplots will be plot per State (and not per Code). 
# For different analysis the code might as well need to be adapted.


def plothistograms(numr, numc, datatoplot: pd.pandas.core.frame.DataFrame, setylim = False, 
                 ymin = 0, ymax = 1, figsizeproj = (18, 11), ylabelplot = "P (mm)", 
                 Cluster = "Cluster",Descriptor = "Descriptor", 
                 font_size_plot = {'font.size': 12}):
    
    """
    Inputs
    ------------------
    numr = Number of rows of your figure;
    numc = Numer of columns of your figure;
    datatoplot: dataframe[Index = Codes; columns = [Cluster, Statistical descriptor]]: 

    setylim = It is used when one needs to set a common y-lim for the graphs;
    ymin and ymax = only used when "setylim" is "True";
    figsizeproj = size of the generated figure in inches;
    ylabelplot = label of the time-series (assuming all time series are in the same units and type);
    
    Returns
    --------------------
    plt.plot: Boxplot. 
        
    """   
    fig, axs = plt.subplots(int(numr),int(numc), figsize = figsizeproj)

    i = 0
    
    for col in datatoplot[Cluster].unique():
    
        plot_data = datatoplot[datatoplot[Cluster] == col].loc[:,Descriptor]
    
        name = col
        
        num_rows = axs.shape[0]
        num_cols = axs.shape[1]

        
        colauxs = [i for i in range(num_cols) for _ in range(num_rows)] 
        rowauxs = [*range(num_rows)] * num_cols
    
        colaux, rowaux = colauxs[i], rowauxs[i]
        
        # Here we can plot some text in our boxplot plots: 
        text_to_plot = "Number: " + str(len(plot_data))
        axs[rowaux,colaux].hist(plot_data)
        axs[rowaux,colaux].set_title(name, loc='left')
        axs[rowaux,colaux].text(0.75, 0.90, text_to_plot, horizontalalignment='center', 
                                verticalalignment='center', transform= axs[rowaux,colaux].transAxes)
        
        axs[rowaux,colaux].set_xlim(0, 100)
        
        
        if setylim == True:
            axs[rowaux,colaux].set_ylim(ymin, ymax)
        else:
            pass
        
        if colaux == 0:
            axs[rowaux,colaux].set_ylabel(ylabelplot)
        
        
        i = i + 1

    plt.rcParams.update(font_size_plot)
    plt.tight_layout()

    return fig, axs

#%% 12. Define a new code for the columns of a dataframe:

# This function is useful for the quick creation of a new unique code for a network and to rename their respective time-series columns 
# in an easy and straight forward way


def new_code_function(network_country_input, timeseries_country_input, name_col_in_network = "code", country = "PT"):
    # This simple function just automatizes the process of defining a new code for our time_series and network:
    
    # name_col_in_network is the name of the column present in the network that matches the columns present in the timeseries
    
    # First we make a copy of our dataframe:
    network_country = network_country_input.copy()
    timeseries_country = timeseries_country_input.copy()
    
    # Now we create a new code column:
    network_country["new_code"] = np.nan
    
    # Now we define a range (1 to ...):
    network_country.loc[:, 'new_code'] = range(1, len(network_country) + 1)
    # Now we convert this range to int to get rid of any decimals:
    network_country.loc[:, 'new_code'] = network_country.loc[:, 'new_code'].astype(int)
    # Now we fill so we can have tge same number of elements regardeles of the number:
    network_country.new_code = network_country.new_code.astype(str).str.zfill(5)
    # And here we set the country code before the new code number:
    network_country.new_code = country + network_country.new_code
    
    # Now to change the timeseries columns:
    # Create a dictionary mapping codes to new column names
    code_to_column = dict(zip(network_country.loc[:, name_col_in_network] , network_country.loc[:, 'new_code'] ))

    # Method 1: Using map()
    timeseries_country.columns = timeseries_country.columns.map(code_to_column)

    return network_country, timeseries_country
    
#%% 13. This function generate a dataframe with the number of measurements at the daily, monthly and annual time-steps:
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

#%% 14. Compute the longest gap periods for each column (measurement) in the input DataFrame.

from tqdm import tqdm

def longest_gap_measurements(timeseries: pd.DataFrame):
    """
    Inputs
    ------------------
    timeseries: dataset[Index = Datetime; columns = [Measurements]]: 
        dataframe with datetime as the index, and with each column representing one measurement. 
        It assumes that the gaps in the measurements are stored as np.nan
    Returns
    --------------------
    pandas.Series with index as column names and values as the longest gap period.
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

#%% 15. Plot a map of points with color-coded categories based on measurement data.
### For example, you can plot the number of years with measurement in a color based map. 

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
        group.plot(ax=ax, color=color_mapping[category], markersize=markersize_map, legend=False, label=category)

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
    mpl.rcParams['font.size'] = 18  # You can adjust this value as needed

#%% 16. Plot a map of points in a map (update from ploitpointsmap).
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
