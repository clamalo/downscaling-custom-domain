from ecmwf.opendata import Client
import matplotlib.pyplot as plt
import matplotlib as mpl
import xarray as xr
import requests
import urllib.request
import time
from datetime import datetime, timedelta
from calendar import monthrange
import numpy as np
import os
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import cartopy.crs as ccrs
import cartopy.feature
import cartopy
from metpy.plots import USCOUNTIES
import multiprocessing
import matplotlib.colors as colors
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from urllib.request import urlopen
from bs4 import BeautifulSoup

#changeable inputs
domains = ['pnw','colorado','utah','norcal','northeast','custom_domain','japan']
max_frame = 240


#FUNCTION: determine the date and time from which to pull data from
def datestr_and_cycle():
    #pull current year, month, day, and hour in UTC time
    datestr = str(datetime.utcnow())
    datestr = datestr.split('-')
    year = str(datestr[0])
    month = str(datestr[1])
    day = str((datestr[2].split(' '))[0])
    datestr = str(datestr[2].split(' ')[1])
    hour = int(datestr[0:2])
    #datestr to pull from NOMADS
    datestr = str(year+month+day)

    #logic checks to make sure it's not pulling data before it's done on NOMADS
    if 8<=hour<20:
        cycle = '00'
    elif 20<=hour or 8>hour:
        cycle = '12'
        #if it's early but not in time for 0z, subtract 1 day and create new datestr
        if 8>hour:
            datestr = datetime.strptime(datestr, '%Y%m%d')
            days = timedelta(1)
            datestr = str(datestr-days)
            datestr = datestr.split('-')
            year = str(datestr[0])
            month = str(datestr[1])
            day = str((datestr[2].split(' '))[0])
            datestr = str(datestr[2].split(' ')[1])
            datestr = str(year+month+day)
        else:
            datestr = datestr
    return datestr,cycle,hour

#set datestr and cycle
datestr = datestr_and_cycle()[0]
cycle = datestr_and_cycle()[1]

#FUNCTION: create list of coordinates for dataset bounding box dimensions based on domain input
def domain_select(domain):
    if domain == 'pnw':
        c = [50.39249,-125.75380,44.94667,-115.14590]
    elif domain == 'utah':
        c = [42.25,-113.25,39,-108.75]
    elif domain == 'norcal':
        c = [39.49244,-123.08545,37.09472,-118.46558]
    elif domain == 'colorado':
        c = [41,-109,36.33170,-104.08140]
    elif domain == 'northeast':
        c = [45.5,-76.75,41.25,-69.5]
    elif domain == 'japan':
        c = [37.57047,136.80028,36.12580,139.02301]
    elif domain == 'custom_domain':
        uf = urllib.request.urlopen('https://clamalo.github.io/downscaling-custom-domain/custom_domain.html')
        custom_domain_list = str(uf.read())
        custom_domain_list = ((custom_domain_list.split('[')[1]).split(']')[0]).split(',')
        c = []
        for coordinate in custom_domain_list:
            c.append(float(coordinate))
    if domain == 'japan':
        max_mins = [(c[0]),(c[2]),(c[3]),(c[1])]
    else:
        max_mins = [(c[0]),(c[2]),(c[3]+360),(c[1]+360)]
    return max_mins


#FUNCTION: make frame as a string in the correct format for NOMADS request
def name_frame(frame):
    if len(str(frame)) == 1:
        frame = '00'+str(frame)
    elif len(str(frame)) == 2:
        frame = '0'+str(frame)
    else:
        frame = str(frame)
    return frame


#FUNCTION: crop any given dataset to bounding dimensions set by domain_select
def crop(ds,domain):
    max_mins = domain_select(domain)
    max_lat = float(max_mins[0])
    min_lat = float(max_mins[1])
    max_lon = float(max_mins[2])
    min_lon = float(max_mins[3])

    mask_lon = (ds.longitude >= min_lon) & (ds.longitude <= max_lon)
    mask_lat = (ds.latitude >= min_lat) & (ds.latitude <= max_lat)
    ds = ds.where(mask_lat, drop=True)
    ds = ds.where(mask_lon, drop=True)
    return ds


#FUNCTION: load downscale NetCDF file and crop it
def trim_ratio(max_mins,resolution,domain):
    if resolution == '40':
        resolution = '4'

    currentYear = datetime.utcnow().year
    currentMonth = datetime.utcnow().month
    currentDay = datetime.utcnow().day

    #DEFINE NUMBER OF DAYS IN THE MONTH
    num_days_current = monthrange(currentYear, currentMonth)[1]
    if currentMonth != 1:
        num_days_last = monthrange(currentYear, currentMonth-1)[1]
    else:
        num_days_last = monthrange(currentYear-1, 12)[1]
    if currentMonth != 12:
        num_days_next = monthrange(currentYear, currentMonth+1)[1]
    else:
        num_days_next = monthrange(currentYear+1, 1)[1]

    #CALCULATE MIDDLE DATE (WHERE THE PRATIOS ARE CENTERED)
    middle_date_current = int(num_days_current/2)+1
    middle_date_last = int(num_days_last/2)+1
    middle_date_next = int(num_days_next/2)+1

    if currentDay < middle_date_current:
        current_percentage = (currentDay+(num_days_last-middle_date_last))/(middle_date_current+(num_days_last-middle_date_last))
        other_month = currentMonth-1
        if other_month == 0:
            other_month = 12
    elif currentDay > middle_date_current:
        current_percentage = 1-((currentDay-middle_date_current)/((num_days_current-middle_date_current)+middle_date_next))
        other_month = currentMonth+1
        if other_month == 13:
            other_month = 1
    elif currentDay == middle_date_current:
        current_percentage = 1
        other_month = currentMonth


    if domain != 'japan':
        current_month_ds = xr.load_dataset('/Users/shedprinter/desktop/downscale_files/p'+resolution+'/us_chelsa_0.'+resolution+'_'+str(currentMonth)+'_area.nc')
        current_month_ds.coords['lon'] = current_month_ds.lon + 360
        current_month_ds = current_month_ds.swap_dims({'lon': 'lon'})
        other_month_ds = xr.load_dataset('/Users/shedprinter/desktop/downscale_files/p'+resolution+'/us_chelsa_0.'+resolution+'_'+str(other_month)+'_area.nc')
        other_month_ds.coords['lon'] = other_month_ds.lon + 360
        other_month_ds = other_month_ds.swap_dims({'lon': 'lon'})
    else:
        current_month_ds = xr.load_dataset('/Users/shedprinter/desktop/downscale_files/p'+resolution+'/jp_chelsa_0.'+resolution+'_'+str(currentMonth)+'_area.nc')
        other_month_ds = xr.load_dataset('/Users/shedprinter/desktop/downscale_files/p'+resolution+'/jp_chelsa_0.'+resolution+'_'+str(other_month)+'_area.nc')

    ds = current_month_ds
    ds['latitude'] = ds['lat']
    ds['longitude'] = ds['lon']
    ds = crop(ds,domain)
    current_month_ds['latitude'] = current_month_ds['lat']
    current_month_ds['longitude'] = current_month_ds['lon']
    current_month_ds = crop(current_month_ds,domain)
    other_month_ds['latitude'] = other_month_ds['lat']
    other_month_ds['longitude'] = other_month_ds['lon']
    other_month_ds = crop(other_month_ds,domain)

    for x in range(len(ds.lat)):
        current_month_list = current_month_ds.pratio[x].values
        other_month_list = other_month_ds.pratio[x].values
        downscaled_output = []
        for current, other in zip(current_month_list, other_month_list):
               if str(current) == 'nan':
                   current=1
               if str(other) == 'nan':
                   other=1
               output = (current_percentage*current)+((1-current_percentage)*other)
               downscaled_output.append(output)
        ds.pratio[x] = downscaled_output

    return ds


#FUNCTION: ingest each grib file from NOMADS and save them to gribs directory
def ingest_gribs(i):
    i = i*6
    print(i)
    frame = name_frame(i)
    datestr = datestr_and_cycle()[0]
    cycle = datestr_and_cycle()[1]
    gfs_url = 'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.t'+cycle+'z.pgrb2.0p25.f'+str(frame)+'&lev_surface=on&var_APCP=on&leftlon=0&rightlon=360&toplat=90&bottomlat=-90&dir=%2Fgfs.'+datestr+'%2F'+cycle+'%2Fatmos'
    r = requests.get(gfs_url, allow_redirects=True)
    open('/Users/shedprinter/desktop/blend_outputs/raw_gribs/'+str(frame)+'gfs.grib2', 'wb').write(r.content)

    gem_url = 'https://dd.weather.gc.ca/model_gem_global/15km/grib2/lat_lon/'+cycle+'/'+str(frame)+'/CMC_glb_APCP_SFC_0_latlon.15x.15_'+datestr+cycle+'_P'+str(frame)+'.grib2'
    r = requests.get(gem_url, allow_redirects=True)
    open('/Users/shedprinter/desktop/blend_outputs/raw_gribs/'+str(frame)+'gem.grib2', 'wb').write(r.content)

    client = Client("ecmwf", beta=True)

    parameters = ['tp']
    filename = 'ecmwf_tp.grib'
    hour = datestr_and_cycle()[2]
    if 8>hour:
        date = -1
    else:
        date = 0

    if int(frame) != 0:
        client.retrieve(
            date=date,
            time=int(cycle),
            step=[int(frame)-6,int(frame)],
            stream="oper",
            type="fc",
            levtype="sfc",
            param=parameters,
            target='/Users/shedprinter/desktop/blend_outputs/raw_gribs/'+str(frame)+'ecmwf.grib'
        )


def blend_models(frame,domain):
    #load raw gribs
    gfs_ds = (xr.load_dataset('/Users/shedprinter/desktop/blend_outputs/raw_gribs/'+frame+'gfs.grib2'))
    gfs_ds['tp'] = gfs_ds['tp']*0.039370079003585

    ecmwf_ds = (xr.load_dataset('/Users/shedprinter/desktop/blend_outputs/raw_gribs/'+frame+'ecmwf.grib',engine='cfgrib'))

    #rename glitchy parameter name
    ecmwf_ds['tp'] = ecmwf_ds['unknown']
    ecmwf_ds = ecmwf_ds.drop(['unknown'])

    #get 6-hourly precip in inches
    ecmwf_ds['tp'] = ((ecmwf_ds.tp[1]*39.370079003585)-(ecmwf_ds.tp[0]*39.370079003585))
    ecmwf_ds = ecmwf_ds.drop_dims('step')

    longitude_list = []

    if domain != 'japan':
        for x in range (len(ecmwf_ds.longitude)):
            longitude = ecmwf_ds.longitude[x]
            longitude = longitude+360
            longitude_list.append(longitude)
        ecmwf_ds['longitude'] = longitude_list

    if int(frame) > 6:
        gem_ds = xr.load_dataset('/Users/shedprinter/desktop/blend_outputs/raw_gribs/'+str(frame)+'gem.grib2', engine='cfgrib')
        gem_ds['tp'] = gem_ds['unknown']

        previous_gem_ds = xr.load_dataset('/Users/shedprinter/desktop/blend_outputs/raw_gribs/'+name_frame(int(frame)-6)+'gem.grib2', engine='cfgrib')
        previous_gem_ds['tp'] = previous_gem_ds['unknown']

        gem_ds['tp'] = (gem_ds['unknown']-previous_gem_ds['unknown'])*0.039370079003585
    else:
        gem_ds = xr.load_dataset('/Users/shedprinter/desktop/blend_outputs/raw_gribs/'+str(frame)+'gem.grib2', engine='cfgrib')
        gem_ds['tp'] = gem_ds['unknown']*0.039370079003585
        gem_ds = gem_ds.drop(['unknown'])

    longitude_list = []

    if domain != 'japan':
        for x in range (len(gem_ds.longitude)):
            longitude = gem_ds.longitude[x]
            longitude = longitude+360
            longitude_list.append(longitude)
        gem_ds['longitude'] = longitude_list

    max_mins = domain_select(domain)
    p25_ds = trim_ratio(max_mins,'25',domain)
    p4_ds = trim_ratio(max_mins,'40',domain)
    p15_ds = trim_ratio(max_mins,'15',domain)

    gfs_ds = gfs_ds.interp(latitude=p25_ds["lat"], longitude=p25_ds["lon"])
    ecmwf_ds = ecmwf_ds.interp(latitude=p25_ds["lat"], longitude=p25_ds["lon"])
    gem_ds = gem_ds.interp(latitude=p25_ds["lat"], longitude=p25_ds["lon"])

    ecmwf_ds = downscale(ecmwf_ds,p4_ds)

    gfs_ds = downscale(gfs_ds,p25_ds)

    gem_ds = downscale(gem_ds,p15_ds)
    #open_and_plot(gem_ds,frame)

    gfs_ds['gfs'] = gfs_ds['tp']*0.000001
    gfs_ds['euro'] = gfs_ds['tp']*0.000001
    gfs_ds['gem'] = gfs_ds['tp']*0.000001
    ds = gfs_ds

    for x in range(len(ecmwf_ds.latitude)):
        gfs_data = gfs_ds.tp[x].values
        ecmwf_data = ecmwf_ds.tp[x].values
        gem_data = gem_ds.tp[x].values

        blended_output = []
        gfs_output = []
        ecmwf_output = []
        gem_output = []

        for gfs_tp, ecmwf_tp, gem_tp in zip(gfs_data, ecmwf_data, gem_data):
            blended_output.append((gfs_tp+ecmwf_tp+gem_tp)/3)
            gfs_output.append(gfs_tp)
            ecmwf_output.append(ecmwf_tp)
            gem_output.append(gem_tp)

        ds.tp[x] = blended_output
        ds['gfs'][x] = gfs_output
        ds['euro'][x] = ecmwf_output
        ds['gem'][x] = gem_output


        ds['tp'][x] = blended_output
        ds['gfs'][x] = gfs_output
        ds['euro'][x] = ecmwf_output
        ds['gem'][x] = gem_output

    return ds

#FUNCTION: access grib from gribs directory, crop it, interpolate it to match ratio_ds dimensions, and downscale it (multiply all values in grib and pratios)
def create_frame_ds(frame,domain):
    #load from gribs directory, open as xarray dataset
    grib_path = '/Users/shedprinter/desktop/blend_outputs/raw_gribs/'+frame+'gfs.grib2'
    ds = xr.load_dataset(grib_path,engine="cfgrib")
    #crop the dataset
    crop(ds,domain)
    #open and crop a new ratio_ds
    ratio_ds = trim_ratio(max_mins,'25',domain)
    #match dimensions of grib dataset to ratio_ds dimensions (interpolation)
    ds.load()
    ds = ds.interp(latitude=ratio_ds["lat"], longitude=ratio_ds["lon"])
    ds = downscale(ds,ratio_ds)
    return ds


#FUNCTION: interpolate dataset
def interpolate(ds):
    new_lon = np.linspace(ds.lon[0], ds.lon[-1], ds.dims["lon"] * 3)
    new_lat = np.linspace(ds.lat[0], ds.lat[-1], ds.dims["lat"] * 3)
    ds = ds.interp(lat=new_lat, lon=new_lon)
    return ds


#FUNCTION: downscale the grib data using the pratios in ratio_ds
def downscale(ds,ratio_ds):
    #for each latitude
    for x in range(len(ds.latitude)):
        #create lists of precip and pratios for that latitude (remember, the datasets are matched in extent and resolution)
        grib_list = ds.tp[x].values
        #conv_list = ds.acpcp[x].values
        ratio_list = ratio_ds.pratio[x].values
        #set empty list each latitude to store downscaled values
        downscaled_output = []
        #for precip, pratio:
        for tp, pratio in zip(grib_list, ratio_list):
               #if the data in pratio is empty, pratio is 1 (no downscaling)
               if str(pratio) == 'nan':
                   pratio=1
               #multiply prate and pratio and append as the downscaled output
               output = tp*pratio
               downscaled_output.append(output)
        #reset all values in current latitude in ratio_ds as the downscaled output
        ds.tp[x] = downscaled_output
    return ds


#FUNCTION: append downscaled output for given frame to accum_ds (total accumulated precip so far)
def append_frame(ds,accum_ds):

    accum_ds['tp'] = accum_ds['tp']+ds['tp']
    accum_ds['gfs'] = accum_ds['gfs']+ds['gfs']
    accum_ds['euro'] = accum_ds['euro']+ds['euro']
    accum_ds['gem'] = accum_ds['gem']+ds['gem']

    return accum_ds


#create the colormap
def create_colormap():
    import matplotlib.colors
    cmap = colors.ListedColormap(['#ffffff', '#bdbfbd','#aba5a5', '#838383', '#6e6e6e', '#b5fbab', '#95f58b','#78f572', '#50f150','#1eb51e', '#0ea10e', '#1464d3', '#2883f1', '#50a5f5','#97d3fb', '#b5f1fb','#fffbab', '#ffe978', '#ffc13c', '#ffa100', '#ff6000','#ff3200', '#e11400','#c10000', '#a50000', '#870000', '#643c31', '#8d6558','#b58d83', '#c7a095','#f1ddd3', '#cecbdc'])#, '#aca0c7', '#9b89bd', '#725ca3','#695294', '#770077','#8d008d', '#b200b2', '#c400c4', '#db00db'])
    return cmap


#FUNCTION: find nearest value in numpy array
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


#FUNCTION: plot time series point forecast plots
def time_series_plots():
    product_types = ['hourly','accumulated']
    for product_type in product_types:
        url = "https://clamalo.github.io/downscaling-custom-domain/points.html"
        html = urlopen(url).read()
        soup = BeautifulSoup(html, features="html.parser")
        text = soup.get_text()
        points = text.split(';')
        points.pop()
        for point in points:
            if str(point.split(',')[0]) == 'Alta':
                point_name = str(point.split(',')[0])
            else:
                point_name = str(point.split(',')[0])[1:]
            domain = str(point.split(',')[3])
            lat = float(point.split(',')[1])
            if domain == 'japan':
                lon = float(point.split(',')[2])
            else:
                lon = float(point.split(',')[2])+360

            print(lat,lon)
            #set empty list to store values
            mean_values = []
            gfs_values = []
            euro_values = []
            gem_values = []
            #for each frame
            for x in range(1,(max_frame+1)):
                #frame naming
                frame = name_frame(x*6)
                #pull correct downscaled dataset (based on hourly vs. accumulated, domain, and frame)
                ds_path = '/Users/shedprinter/desktop/blend_outputs/downscaled_gribs/'+product_type+'/'+domain+'/'+frame+'downscaled.nc'
                ds = xr.load_dataset(ds_path)
                #pull value at given coordinates
                closest_lat = find_nearest(ds.lat.values, lat)
                closest_lon = find_nearest(ds.lon.values, lon)
                mean_value = float(ds.tp[closest_lat][closest_lon])
                gfs_value = float(ds.gfs[closest_lat][closest_lon])
                euro_value = float(ds.euro[closest_lat][closest_lon])
                gem_value = float(ds.gem[closest_lat][closest_lon])
                #print(euro_value,'euro')
                if str(mean_value) == 'nan':
                    mean_value = 0
                if str(gfs_value) == 'nan':
                    gfs_value = 0
                if str(euro_value) == 'nan':
                    euro_value = 0
                if str(gem_value) == 'nan':
                    gem_value = 0
                #append value to list of values by hour
                mean_values.append(mean_value)
                gfs_values.append(gfs_value)
                euro_values.append(euro_value)
                gem_values.append(gem_value)
            #reset plot figure size and ax zoom
            fig = plt.figure(figsize=(6.5, 4))
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])

            #create initialization time and valid time for given frame for plot title
            datestr = datestr_and_cycle()[0]
            cycle = datestr_and_cycle()[1]
            init_label = datestr[0:4]+'-'+datestr[4:6]+'-'+datestr[6:8]+' '+cycle+'z'

            #plot title based on product
            if product_type == 'accumulated':
                plt.title(point_name+" Downscaled Accumulated Precipitation (6-Hourly) || Init "+init_label,fontsize=7)
            elif product_type == 'hourly':
                plt.title(point_name+" Downscaled 6-Hourly Precipitation Rate || Init "+init_label,fontsize=7)
            #x and y axis titles
            ax.set_xlabel("Forecast Hour",fontsize=6)
            ax.set_ylabel("Precipitation (Inches)",fontsize=6)
            plt.yticks(fontsize=5)
            #x and y min/max
            ax.set_xlim([0, max_frame-1])

            max_tick = float(round(max(mean_values+gfs_values+euro_values+gem_values),1)+.1)
            ax.set_ylim([0, max_tick])
            #set x ticks
            ax.set_xticks(np.arange(0, max_frame, 4.0))
            plt.xticks(rotation = 45)
            plt.xticks(fontsize=4)
            #x tick naming (since hour 0 is actually forecast hour 1)
            tick_label_list = []

            for x in range(1,max_frame+1,4):
                datestr = datestr_and_cycle()[0]
                cycle = datestr_and_cycle()[1]
                datestr = str(datestr)+str(cycle)
                datestr = datetime.strptime(datestr, '%Y%m%d%H')
                hours_added = timedelta(hours = int(x*6))
                datestr = str(datestr+hours_added)
                datestr = datestr.split('-')
                datestr = (datestr[0]+datestr[1]+datestr[2]).split(' ')
                datestr = str(datestr[0]+(datestr[1]).split(':')[0])
                datestr = datestr[4:6]+'/'+datestr[6:8]+' '+datestr[8:10]+'z'

                tick_label_list.append(datestr)

            ax.set_xticklabels(tick_label_list)
            #create grid on background
            ax.grid()
            #plot list of values
            ax.plot(mean_values, label='Mean', alpha=1)
            ax.plot(gfs_values, label = 'GFS', alpha=0.25)
            ax.plot(euro_values, label = 'ECMWF', alpha=0.25)
            ax.plot(gem_values, label = 'GDPS', alpha=0.25)
            plt.legend(prop={"size":7})
            #save figure to correct directory
            plt.savefig('/Users/shedprinter/desktop/blend_outputs/images/pointforecasts/'+product_type+'/'+domain+'/'+point_name+'_plot.png',dpi=300)


#FUNCTION: plot the output in matplotlib
def plot_data(i):
    if i != 0:
        i = i*6
        product_types = ['hourly','accumulated']
        #script only plots accumulated data
        #product_types = ['accumulated']
        for product_type in product_types:
            frame = name_frame(i)
            for domain in domains:

                url = "https://clamalo.github.io/downscaling-custom-domain/points.html"
                html = urlopen(url).read()
                soup = BeautifulSoup(html, features="html.parser")
                text = soup.get_text()
                points = text.split(';')
                points.pop()

                point_lats = []
                point_lons = []

                for point in points:
                    print(point)
                    point_domain = str(point.split(',')[3])
                    if point_domain == domain:
                        lat = float(point.split(',')[1])
                        if domain == 'japan':
                            lon = float(point.split(',')[2])
                        else:
                            lon = float(point.split(',')[2])+360
                        point_lats.append(lat)
                        point_lons.append(lon)

                print(domain,i)
                datestr = datestr_and_cycle()[0]
                grib_path = '/Users/shedprinter/desktop/blend_outputs/downscaled_gribs/'+product_type+'/'+domain+'/'+frame+'downscaled.nc'
                dataset = xr.load_dataset(grib_path)
                #set font size and line width in matplotlib
                mpl.rcParams.update({'font.size': 10})
                mpl.rcParams['axes.linewidth'] = 1
                #initialize "shorthand" variables for data
                precip = dataset.variables['tp']
                lats = dataset.variables['lat'][:]
                lons = dataset.variables['lon'][:]
                #set plot figure size
                fig = plt.figure(figsize=(12, 8))
                #cartopy map projection type
                ax = plt.axes(projection=ccrs.PlateCarree())
                #numpy linspace for colorbar extent and increments (min,max,increments)
                us = np.linspace(0, 4, 1000, endpoint=True)
                #create contour plot on ax
                newcmp = create_colormap()
                bounds = [0,0.01,0.03,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.2,1.4,1.6,1.8,2,2.5,3,3.5,4,5,6,7,8,9,10]#,11,12,13,14,15,16,17,18,19,20])
                #norm = colors.BoundaryNorm(boundaries=bounds, ncolors=41)
                norm = colors.BoundaryNorm(boundaries=bounds, ncolors=32)
                #norm = colors.BoundaryNorm(bounds, newcmp.N, clip=True)
                #cf = ax.contourf(lons,lats,precip,us,norm=norm,cmap=newcmp)
                cf = ax.pcolormesh(lons, lats, precip, norm=norm, cmap=newcmp)
                cd = ax.scatter(x=point_lons, y=point_lats, s=7, color='#000000', alpha=1)
                #set basemap features using cartopy
                if domain == 'northeast':
                    ax.add_feature(cartopy.feature.STATES,linewidth=1)
                else:
                    ax.add_feature(USCOUNTIES.with_scale('500k'),linewidth=1)
                    #ax.add_feature(cartopy.feature.STATES)
                    ax.add_feature(cartopy.feature.COASTLINE)

                #create initialization time and valid time for given frame for plot title
                datestr = datestr_and_cycle()[0]
                cycle = datestr_and_cycle()[1]
                init_label = datestr[0:4]+'-'+datestr[4:6]+'-'+datestr[6:8]+' '+cycle+'z'
                datestr = str(datestr)+str(cycle)
                datestr = datetime.strptime(datestr, '%Y%m%d%H')
                hours_added = timedelta(hours = int(frame))
                datestr = str(datestr+hours_added)
                valid_label = datestr[0:4]+'-'+datestr[5:7]+'-'+datestr[8:13]+'z'

                #set plot title
                plt.title("Downscaled Accumulated Precipitation (Inches) || Forecast Hour "+str(frame)+" || Init "+init_label+" || Valid "+valid_label,fontsize=10)
                #set x and y labels
                ax.set_xlabel("Degrees Longitude")
                ax.set_ylabel("Degrees Latitude")
                #create the colorbar, shrink it so it fits
                cbar = plt.colorbar(cf, shrink=0.7, orientation="horizontal", pad=0.03)
                #set tick locations and labels for the colorbar
                cbar.set_ticks([0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.2, 1.6, 2, 3, 4, 6, 8, 10])
                cbar.set_ticklabels(['0.01', '0.05', '0.1', '0.2', '0.3', '0.5', '0.7', '0.9', '1.2', '1.6', '2', '3', '4', '6', '8', '10'])
                cbar.ax.tick_params(labelsize=8)
                cbar.ax.tick_params(width=0.25)
                #save image to correct path based on domain
                path = str('/Users/shedprinter/desktop/blend_outputs/images/'+product_type+'/'+domain+'/')
                plt.savefig(path+str(frame)+'image.png',dpi=500,bbox_inches='tight')
                #remove colorbar and figure (otherwise it messes up the next frame)
                cbar.remove()
                #ax.cla()
                plt.clf()


#FUNCTION: push outputs to github
def github_push():
    os.chdir('/Users/shedprinter/desktop/blend_outputs/')
    os.system('git add images')
    os.system('git add model/pointforecasts.html')
    print('files added')
    os.system('git commit -m "auto-push"')
    print('commit initialized')
    #os.system('git stash')
    #os.system('git stash push --include-untracked')
    os.system('git checkout master')
    os.system('git pull')
    os.system('git config --global core.askpass "git-gui--askpass"')
    os.system('git push')
    #print('pushed to github')


def create_hourly_frame(i):
    domains = ['pnw','colorado','utah','norcal','northeast','custom_domain','japan']
    if i != 0:
        for domain in domains:
            frame = name_frame(i*6)
            print(domain+' processing frame '+str(int(frame))+'...')
            frame = name_frame(frame)
            #create ratio_ds (grib data for current frame) and interpolate
            ds = blend_models(frame,domain)
            #ratio_ds = create_frame_ds(frame)
            ds = interpolate(ds)
            #save hourly dataset as netcdf
            ds.to_netcdf('/Users/shedprinter/desktop/blend_outputs/downscaled_gribs/hourly/'+domain+'/'+frame+'downscaled.nc', 'w')

def create_html_file():
    url = "https://clamalo.github.io/downscaling-custom-domain/points.html"
    html = urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")
    text = soup.get_text()
    points = text.split(';')
    points.pop()

    # to open/create a new html file in the write mode
    f = open('/Users/shedprinter/desktop/blend_outputs/model/pointforecasts.html', 'w')

    # the html code which will go in the file GFG.html
    html_template = """<html>
    <head>
    <title>Downscaled point forecasts</title>
    </head>
    <body>"""
    for domain in domains:
        html_template+=('<h1>'+domain+'</h1>')
        for point in points:
            if str(point.split(',')[0]) == 'Alta':
                point_name = str(point.split(',')[0])
            else:
                point_name = str(point.split(',')[0])[1:]
            point_domain = str(point.split(',')[3])
            if point_domain == domain:
                html_template+=('<h3>'+point_name+'<a href=/images/pointforecasts/hourly/'+point_domain+'/'+point_name+'_plot.png> (Hourly)</a> <a href=/images/pointforecasts/accumulated/'+point_domain+'/'+point_name+'_plot.png> (Accumulated)</a></h3>\n')

    """
    </body>
    </html>
    """
    # writing the code into the file
    f.write(html_template)
    f.close()

if __name__ == '__main__':
    create_html_file()
    #start timer
    start = time.time()

    #set initial frame to 1, name it correctly
    frame = name_frame(1)

    #multiprocessing code
    p = multiprocessing.Pool(10)
    p.map(ingest_gribs, range(int(max_frame/6)+1))

    #for each domain
    for domain in domains:
        #select coordinates for bounding boxes of datasets
        max_mins = domain_select(domain)
        #each cycle through domain, reset frame to 1
        frame = name_frame(6)
        #create initial ratio_ds for accum_ds ("template" dataset) and interpolate
        accum_ds = blend_models(frame,domain)

        accum_ds['tp'] = accum_ds['tp']*0
        accum_ds['gfs'] = accum_ds['gfs']*0
        accum_ds['euro'] = accum_ds['euro']*0
        accum_ds['gem'] = accum_ds['gem']*0

        accum_ds = interpolate(accum_ds)
        accum_ds.to_netcdf('/Users/shedprinter/desktop/blend_outputs/downscaled_gribs/accumulated/'+domain+'/'+'accum_ds.nc')
        #set next frame
        frame = name_frame(6)

    p = multiprocessing.Pool(3)
    p.map(create_hourly_frame, range(int(max_frame/6)+1))

    for domain in domains:
        frame = name_frame(6)
        while int(frame) <= int(max_frame):
            print(frame,domain)
            if int(frame) == 6:
                accum_ds = xr.load_dataset('/Users/shedprinter/desktop/blend_outputs/downscaled_gribs/accumulated/'+domain+'/'+'accum_ds.nc')
            #add to accum_ds (total accumulated precip)
            ds = xr.load_dataset('/Users/shedprinter/desktop/blend_outputs/downscaled_gribs/hourly/'+domain+'/'+frame+'downscaled.nc')
            accum_ds = append_frame(ds,accum_ds)
            #save accumulated ds as netcdf
            accum_ds.to_netcdf('/Users/shedprinter/desktop/blend_outputs/downscaled_gribs/accumulated/'+domain+'/'+frame+'downscaled.nc', 'w')
            #go to next frame (and name correctly)
            frame = name_frame(int(frame)+6)


    max_frame = int(max_frame/6)
    #set the new colormap
    newcmp = create_colormap()
    #time_series_plots()

    print('starting multiprocessing')

    #multiprocessing code
    p = multiprocessing.Pool(3)
    p.map(plot_data, range(max_frame+1))
    newend = time.time()

    #github_push()

    end = time.time()
    print(end-start)
