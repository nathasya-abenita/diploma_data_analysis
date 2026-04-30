import numpy as np 
import matplotlib.pyplot as plt
from netCDF4 import Dataset
import matplotlib.ticker as ticker
import urllib.request
import os

def find_changepoint(x, y, order=1):
    rss = []
    buffer=20
    breaklist=list(range(buffer, len(x)- buffer))
    
    # swoop through the series, calculating RSS for each break point    
    # start can't be 2 for the covariance calculation, 
    # let's assume we need at least 10 datapoints anyway to get a reasonable fit
    for n in breaklist:
        _, rssval = piecewise_polyfit(x, y, n, order)
        rss.append(rssval)
    
    # x0 is the onset time from the minimum RSS
    n0 = np.nanargmin(rss)
    brk_point=breaklist[n0]

    x1, x2 = split(x, brk_point) #renamed breakpoint to split_point
    y1, y2 = split(y, brk_point) #renamed breakpoint to split_point
    p1_best = np.polyfit(x1, y1, order)
    p2_best = np.polyfit(x2, y2, order)

    ypred, _ = piecewise_polyfit(x, y, brk_point, order) #renamed breakpoint to split_point

    return brk_point, ypred, rss

def split(x, n):
    # split a series at point n
    return x[:n], x[n:]

def piecewise_polyfit(x, y, n, order=1):
    
    y = np.ma.masked_array(y, np.isnan(y))

    # split x and y at point n 
    x1, x2 = split(x, n)
    y1, y2 = split(y, n)

    # fit nth order (default=1) fit to two parts
    p1 = np.ma.polyfit(x1, y1, order)
    p2 = np.ma.polyfit(x2, y2, order)

    # check for errors:
    if np.isnan(p1).any() or np.isnan(p2).any():
        raise ValueError('NaN for polyfit coeffs. Check data.')

    # make the series based on the two piecewise fits
    ypred1 = np.polyval(p1, x1)
    ypred2 = np.polyval(p2, x2)
    ypred = np.concatenate([ypred1, ypred2])

    # calculate the Mean Square error of the piecewise fit to the
    # original data, and return fit and RSS
    rss = np.sum((y - ypred)**2)
    return ypred, rss

def get_file(region,var,year):
    datadir="http://clima-dods.ictp.it/Users/tompkins/monsoons/data/"
    if region=="india":
        file = var+"_"+str(year)+"_asia_sn10_30_we60_100.nc"
        url=datadir+file
    if region=="nam":
        file = "era5_p_e_vimd_"+str(year)+"_daysum_NAM.nc"
        url=datadir+file
    if region=="cumnam":
        file = "era5_p_e_vimd_"+str(year)+"_daysum_NAM_timcumsum.nc"
        url=datadir+file

    data_dir = './data/monsoon/'
    if not os.path.exists(data_dir):
        print(f'creating data directory: {data_dir}')
        os.mkdir(data_dir)

    path = data_dir + file

    if not os.path.exists(path): # only download if it doesn't exist
        print(f'downloading: {file}')
        try:
            url = (url+'#mode=bytes')
            urllib.request.urlretrieve(url, path)
        except:
            url = url 
            urllib.request.urlretrieve(url, path)
    
    dset = Dataset(path)
    return(dset)

def get_data(dlist,var,fldname,sf=1,year=2000):
    #ds=get_file(var,year)
    region="india"
    ds=get_file(region,var,year)
    fld=np.squeeze(ds[fldname])*sf
    dlist.append({"var":var,"data":fld,"year":year})
    return dlist

# make a simple line plot for the 3 styles
def plotyear(dlist):
    fig,ax = plt.subplots()

    # hard wired 
    lstyles=["dashed","solid","solid"]
    lwidths=[2,1,1]
    lcols=["grey","grey","black"]
    tick_spacing=20
    
    # fudge to correct labels to something nicer
    labels={"evap":"evaporation","prec":"precipitation","vimd":"MFC"}

    for ivar,var in enumerate(dlist):
        ax.plot(var["data"],
                linestyle=lstyles[ivar],
                linewidth=lwidths[ivar],
                color=lcols[ivar],label=labels[var["var"]])

    ax.set_xlabel("day of year")
    ax.set_ylabel("(mm/day)")
    ax.set_title("year "+str(var["year"]))
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    plt.xticks(rotation=-45)
    ax.xaxis.grid(linestyle="dotted") # vertical lines
    ax.legend()
    plt.savefig("precip.jpg",bbox_inches='tight',dpi=300)
    return(fig,ax)

def plot_breakpoint(dlist, cmfc, ypred_best, day_onset):
    fig,ax=plotyear(dlist)

    # add CMFC on rh y axis
    ax2=ax.twinx()
    ax2.plot(cmfc,label="CMFC",color="red")
    ax.set_xlabel("Day of Year")
    ax2.set_ylabel("CMFC (mm)")
    ax2.legend()

    ax2.plot(ypred_best,color="green",linestyle=":",linewidth=2)

    # add vertical line at onset date
    ax.axvline(day_onset,color="blue")
    ax.axvline(day_onset,color="blue")
    ax.text(x=day_onset+3,y=12,s="Onset",color="blue",size=20)
    plt.show()
    

if __name__ == '__main__':

    # Parameters
    region="india" # india, nam...  

    fldnames=["tp","e","vimd"]
    vars=["prec","evap","vimd"]
    sf=[1,-1,1]
    dlist=[]
    year_list = range(1987, 2019+1)

    cutoff=250 # cutoff days onset
    cess_start = 160 # cessation days start
    thr = 5 # mm/d

    # Iterate over years
    onset_list = []
    onset_simple_list = []
    cessation_list = []

    for year in year_list:

        # Cycle through the fields and add each dataset as a dictionary to the list
        dlist=[]
        for ivar,var in enumerate(vars):
            dlist=get_data(dlist,vars[ivar],fldnames[ivar],sf=sf[ivar],year=year)
        
        # First let's get the cmfc, only one field so we we extract it from the list and only keep the data
        cmfc=get_data([],var="cum_vimd",fldname="vimd",year=year)[0]["data"]
        
        # First we define the number of time steps
        ntim=len(cmfc)
        x=list(range(ntim))

        # Compute day onset
        day_onset,ypred_best,rss = find_changepoint(x[0:cutoff],cmfc[0:cutoff])

        # Compute day cessation
        day_cessation,ypred_best,rss = find_changepoint(x[cess_start:],cmfc[cess_start:])

        # Compute simple onset with precipitation
        
        day_onset_simple = np.argmax(dlist[0]['data'] > thr)

        # Save onset
        onset_list.append(day_onset)
        cessation_list.append(day_cessation)
        onset_simple_list.append(day_onset_simple)

    # Plot onset with changepoint vs threshold
    plt.plot(year_list, onset_list, label='Changepoint method')
    plt.plot(year_list, onset_simple_list, label=f'Threshold method ({thr} mm/d)')
    plt.legend()
    plt.xlabel('Year'); plt.ylabel('Number of Day')
    plt.title('Comparing Methods to Define Indian Monsoon Onset (1987-2019)')
    plt.savefig('./output/monsoon_onset_comparison.png')
    plt.close()

    # Turn to numpy array and normalize
    onset_list = np.array(onset_list) - np.mean(onset_list)
    cessation_list = np.array(cessation_list) - np.mean(cessation_list)
    cor = np.corrcoef(onset_list, cessation_list)[0, 1]

    # Plot
    plt.plot(year_list, onset_list, label='Onset')
    plt.plot(year_list, cessation_list, label='Cessation')
    plt.xlabel('Year')
    plt.ylabel('Number of Day Anomalies')
    plt.suptitle('Indian Monsoon Onset and Cessation Days (1987-2019)')
    plt.title(rf'$R={cor:.3f}$')
    plt.legend()
    plt.grid()
    plt.savefig('./output/monsoon_variability.png')
