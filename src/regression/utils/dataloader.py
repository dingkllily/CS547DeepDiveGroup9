import urllib
import pandas as pd
import datetime as dt
import numpy as np


def getfile(location_pair,**kwargs): #tries to get local version and then defaults to google drive version
    (loc,gdrive)=location_pair
    try:
        out=pd.read_csv(loc,**kwargs)
    except FileNotFoundError:
        print("local file not found; accessing Google Drive")
        loc = 'https://drive.google.com/uc?export=download&id='+gdrive.split('/')[-2]
        out= pd.read_csv(loc,**kwargs)
    return out


def getLargeFile(location_pair,**kwargs): #tries to get local version and then defaults to google drive version
    (loc,gdrive)=location_pair
    try:
        out=pd.read_csv(loc,**kwargs)
    except FileNotFoundError:
        print("local file not found; accessing Google Drive")
        url = gdrive
        urllib.request.urlretrieve(gdrive, loc)
        out= pd.read_csv(loc,**kwargs)
    return out

# convert into 4-char strings
def makemil(time):
    ntime = ""
    if len(str(time)) == 1:
        ntime = "000" + str(time)
    if len(str(time)) == 2:
        ntime = "00" + str(time)
    if len(str(time)) == 3:
        ntime = "0" + str(time)
    if len(str(time)) == 4:
        ntime = str(time)
    return ntime

# function for retrieving the hour of the time
def returnhour(miltime):
    return miltime[:2]

def loadCrimeDataset():
    fname = ("Crime_Data_from_2020_to_Present.csv", 
            "https://drive.google.com/u/0/uc?id=1WwLkX_BADtPY83rFQ4CQ5Lbf6UcI055A&export=download&confirm=t&uuid=4e0aae58-2f5b-4420-941f-8f2f49ae0cb8&at=ALAFpqzX_FrSz-hUtqhaXe2ui7h6:1667673013562")
    df = getLargeFile(fname)
    df['Date Rptd'] = df['Date Rptd'].apply(lambda x: dt.datetime.strptime(x, "%m/%d/%Y %H:%M:%S %p").date())
    df['DATE OCC'] = df['DATE OCC'].apply(lambda x: dt.datetime.strptime(x, "%m/%d/%Y %H:%M:%S %p").date())
    df['Date Rptd'] = pd.to_datetime(df['Date Rptd'])
    df['DATE OCC'] = pd.to_datetime(df['DATE OCC'])

    df['Month OCC'] = df['DATE OCC'].dt.month
    df['Year OCC'] = df['DATE OCC'].dt.year
    month_year_occ = [str(m)+"/"+str(y) for m,y in zip(df["Month OCC"], df["Year OCC"])]
    month_year_occ_formatted = [dt.datetime.strptime(d, "%m/%Y") for d in month_year_occ]
    df["Month Year OCC"] = np.array(month_year_occ_formatted)

    df["TIME OCC"] = df["TIME OCC"].apply(makemil)

    df['Hour OCC'] =  df["TIME OCC"].apply(returnhour)

    df['Count'] = np.ones(df.shape[0])

    return df 