# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 13:14:32 2021

@author: Will.Tyree
"""

import pandas as pd
import datetime as dt
import numpy as np

'''
A script that takes in the speed file pulled from the RITIS PDA API, and the
TMC profiles file, and calculates Vehicle Miles Traveled (VMT),
Person Hours of Delay (PHD), Vehicle Hours of Delay (VHD) and User Delay Cost.
Commercial (trucking) and passenger metrics are calculated and then summed to get total PHD/VHD.
'''
class CalculatePerformanceMetrics:
    '''
    Initialize the Calculator object
    '''
    def __init__(self, metrics, profiles):
        self.metrics = metrics
        self.profiles = profiles
    
    '''
    Transform measurement_tstamp into a datetime object and create a time and day name column
    '''
    def setupMetricsTimestamps(self):
        self.metrics['measurement_tstamp'] = pd.to_datetime(self.metrics['measurement_tstamp'])
        self.metrics['Time'] = self.metrics['measurement_tstamp'].dt.time
        self.metrics['Hour'] = self.metrics['measurement_tstamp'].dt.hour
        self.metrics.rename(columns={'tmc_code':'Tmc'}, inplace=True)
        self.metrics['Day'] = self.metrics['measurement_tstamp'].dt.day_name()
        self.metrics['Date'] = self.metrics['measurement_tstamp'].dt.date
        return self.metrics
    
    '''
    Pull the daily volume factor for the 15 minute interval from the TMC profiles data.
    day15: Name of the profiles column that has the volume factors for that day, wuthout the
            15 minute interval number in front of it. ex. 'FifteenVolume0' will get all intervals
            for Sunday
    day: The name of the day, ex. 'Sunday'
    tag: 'Vol15' or 'Truck15' for passenger and commercial volume factors
    '''
    def getDailyVolumes(self, day15, day, tag):
        pd.options.mode.chained_assignment = None  # default='warn'
        vol15cols = [col for col in self.profiles.columns if day15 in col]
        vol15cols.insert(0, 'Tmc')
        vol15 = self.profiles[vol15cols]
        vol15 = pd.melt(vol15, id_vars=vol15cols[0], value_vars=vol15cols[1:])
        vol15.rename(columns={'variable':'Time', 'value':tag}, inplace=True)
        vol15['Day'] = day
        renameVolKeys = vol15cols[1:]
        renameVals = self.metrics['Time'].unique()
        renameVolDict = {renameVolKeys[i]: renameVals[i] for i in range(len(renameVolKeys))}
        vol15.Time.replace(renameVolDict, inplace=True)
        return vol15
    
    '''
    Function that calls the previous function for each day for both commercial and passenger
    and concatenates them into one dataframe and then merges it into the metrics dataframe 
    on the 'Tmc', 'Time', and 'Day' columns. It then adds in more columns from the profiles
    data such as 'District', 'Segment', etc.
    '''
    def concatenateMetricsTable(self):
        voldays = ['FifteenVolume0','FifteenVolume1','FifteenVolume2',
                    'FifteenVolume3','FifteenVolume4','FifteenVolume5','FifteenVolume6']
        volTruckDays = ['FifteenTruck0','FifteenTruck1','FifteenTruck2',
                        'FifteenVolume3','FifteenVolume4','FifteenVolume5','FifteenVolume6']
        days = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
        v, t = [], []
        
        for i in range(0,7):
            vol = self.getDailyVolumes(voldays[i], days[i], 'Vol15')
            v.append(vol)
            truck = self.getDailyVolumes(volTruckDays[i], days[i], 'Comm-Pct')
            t.append(truck)
            
        vol15 = pd.concat(v)
        truck15 = pd.concat(t)
        self.metrics = self.metrics.merge(vol15, on=['Tmc', 'Time', 'Day'])
        self.metrics = self.metrics.merge(truck15, on=['Tmc', 'Time', 'Day'])
        other_cols = ['Tmc', 'District', 'New Route','Combined Seg', 'FFS-14', 'Length-U14', 'vol2019']
        tmc_info = self.profiles[other_cols]
        self.metrics = self.metrics.merge(tmc_info, on=['Tmc'])
        return self.metrics
    
    '''
    Function that calls the above functions in order to preprocess the metrics table 
    prior to calculating the performance metrics
    '''
    def prepareMetricsTable(self):
        self.metrics = self.setupMetricsTimestamps()
        self.metrics = self.concatenateMetricsTable()
        return self.metrics

    '''
    Calculate segment density at the 15 minute interval, true volume, and the month density
    '''
    def calculateDensities(self):
        self.metrics['Density'] = 200/(1+(self.metrics['speed']/15))
        self.metrics['True-Volume'] = self.metrics['vol2019'].astype(int) * self.metrics['Vol15']
        self.metrics['Month-Density'] = self.metrics['True-Volume']/self.metrics['speed']
        return self.metrics
    
    '''
    Various metrics needed to calculate VMT, PHD, VHD
    '''
    def calculateFailures(self):
        self.metrics['Flow'] = (200*self.metrics['speed'])/(1+(self.metrics['speed']/15))
        self.metrics['Trav'] = self.metrics['Flow']*2*0.25
        self.metrics['PropFailed'] = (1-((self.metrics['speed']*.25)/self.metrics['Length-U14']))
        self.metrics.loc[self.metrics['PropFailed'] < 0, 'PropFailed'] = 0
        self.metrics['Failed-Trav'] = self.metrics['Density']*self.metrics['Length-U14']*self.metrics['PropFailed']
        self.metrics['Month-Failed-Trav'] = self.metrics['Month-Density'] * self.metrics['Length-U14'] * self.metrics['PropFailed']
        return self.metrics
    
    '''
    Calculate segment/interval volumes: Limited volume, commercial volume, passenger volume
    '''
    def calculateVolumes(self):
        self.metrics['Limited-Volume'] = np.where(
            self.metrics['True-Volume']+self.metrics['Month-Failed-Trav'] < self.metrics['Trav']+self.metrics['Failed-Trav'],
            self.metrics['True-Volume']+self.metrics['Month-Failed-Trav'],
            self.metrics['Trav']+self.metrics['Failed-Trav']
        )

        self.metrics['Comm-Volume'] = np.where((
            self.metrics['FFS-14'] > 50) & (self.metrics['speed'] < 50),
            self.metrics['Limited-Volume']*(self.metrics['Comm-Pct']/100),
            self.metrics['True-Volume']*(self.metrics['Comm-Pct']/100)
        )

        self.metrics['Pass-Volume'] = np.where((
            self.metrics['FFS-14'] > 50) & (self.metrics['speed'] < 50),
            self.metrics['Limited-Volume']*((100-self.metrics['Comm-Pct'])/100),
            self.metrics['True-Volume']*((100-self.metrics['Comm-Pct'])/100)
        )
        return self.metrics
    
    '''
    Calculate the passenger and commercial Vehicle Miles Traveled
    '''
    def calculateVMTs(self):
        self.metrics['VMT-Comm'] = self.metrics['Comm-Volume']*self.metrics['Length-U14']
        self.metrics['VMT-Pass'] = self.metrics['Pass-Volume']*self.metrics['Length-U14']
        return self.metrics
    
    '''
    Calculate the Passenger Hours of Delay
        If the speed is less than the free flow speed
            PHD equals the VMT divided by the speed at that segment/interval,
            minues the VMT divided by the historic free flow speed at that segment/interval
        Else
            0
    '''
    def calculatePHD(self):
        self.metrics['Ref-PHD-Comm'] = np.where(
            self.metrics['speed'] < self.metrics['FFS-14'],
            (self.metrics['VMT-Comm']/self.metrics['speed'])-(self.metrics['VMT-Comm']/self.metrics['FFS-14']),
            0
        )

        self.metrics['Ref-PHD-Pass'] = np.where(
            self.metrics['speed'] < self.metrics['FFS-14'],
            ((self.metrics['VMT-Pass']/self.metrics['speed'])-(self.metrics['VMT-Pass']/self.metrics['FFS-14']))*1.7,
            0
        )

        self.metrics['Ref-PHD'] = self.metrics['Ref-PHD-Comm']+self.metrics['Ref-PHD-Pass']
        return self.metrics

    '''
    Calculate the Vehicle Hours of Delay
        If the speed is less than the free flow speed minus 20
            VHD equals the VMT divided by the speed at that segment/interval,
            minues the VMT divided by the historic free flow speed minus 20 at that segment/interval
        Else
            0
    '''
    def calculateVHD(self):
        self.metrics['Ref-VHD-Comm-20'] = np.where(
            self.metrics['speed'] < (self.metrics['FFS-14']-20),
            ((self.metrics['VMT-Comm']/self.metrics['speed'])-(self.metrics['VMT-Comm']/(self.metrics['FFS-14']-20))),
            0
        )

        self.metrics['Ref-VHD-Pass-20'] = np.where(
            self.metrics['speed'] < (self.metrics['FFS-14']-20),
            ((self.metrics['VMT-Pass']/self.metrics['speed'])-(self.metrics['VMT-Pass']/(self.metrics['FFS-14']-20))),
            0
        )

        self.metrics['Ref-VHD-20'] = self.metrics['Ref-VHD-Comm-20']+self.metrics['Ref-VHD-Pass-20']
        return self.metrics
    
    '''
    Calculate the User Delay Cost (monetary value associated with VHD)
        Multiply the commercial VHD by a factor of 100.49, the passanger
        VHD by a factor of 17.91, and add them up.
    Factor values are from the University of Washington
    '''
    def calculateDelayCost(self):
        self.metrics['Delay-Cost'] = (self.metrics['Ref-VHD-Comm-20']*100.49)+(self.metrics['Ref-VHD-Pass-20']*17.91)
        return self.metrics
    
    '''
    Condense entire workflow into a single function 
    '''
    def returnFinalMetricsTable(self):
        self.metrics = self.prepareMetricsTable()
        self.metrics = self.calculateDensities()
        self.metrics = self.calculateFailures()
        self.metrics = self.calculateVolumes()
        self.metrics = self.calculateVMTs()
        self.metrics = self.calculatePHD()
        self.metrics = self.calculateVHD()
        self.metrics = self.calculateDelayCost()
        return self.metrics
    
    def getVHDtableInJoshFormat(self):
        self.metrics = self.returnFinalMetricsTable()
        joshVHDtable = self.metrics[['District', 'New Route','Combined Seg', 'Date', 'Hour', 'Ref-VHD-20']]
        joshVHDtable=joshVHDtable.groupby(['District', 'New Route','Combined Seg', 'Date','Hour'])['Ref-VHD-20'].sum().reset_index()
        joshVHDtable = joshVHDtable.pivot(index=['District', 'New Route','Combined Seg', 'Date'], columns="Hour", values="Ref-VHD-20").reset_index()
        joshVHDtable['Date'] = pd.to_datetime(joshVHDtable['Date'])
        joshVHDtable['Date'] = joshVHDtable['Date'].dt.strftime('%B %d,%Y')
        return joshVHDtable