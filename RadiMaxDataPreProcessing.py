import pandas as pd
import numpy as np
from numpy import std
import matplotlib.pyplot as plt

def RL_processing(data_May,data_June,data_July,imageArea,Square_root=False):

    data_May.rename(columns={'Camera':'Camera','tube': 'tube','Depth':'Depth','Date':'Date',data_May.columns[4]:'Root_length'},inplace=True)
    data_June.rename(columns={'Camera':'Camera','tube': 'tube','Depth':'Depth','Date':'Date',data_June.columns[4]:'Root_length'},inplace=True)
    data_July.rename(columns={'Camera':'Camera','tube': 'tube','Depth':'Depth','Date':'Date',data_July.columns[4]:'Root_length'},inplace=True)

    data_May.replace(' NA', np.nan,inplace=True)
    data_June.replace(' NA', np.nan,inplace=True)
    data_July.replace(' NA', np.nan,inplace=True)

    data_May['Root_length']=pd.to_numeric(data_May['Root_length'])
    data_June['Root_length']=pd.to_numeric(data_June['Root_length'])
    data_July['Root_length']=pd.to_numeric(data_July['Root_length'])

    ###Convert estimated root length from pixel length to actual length using calibration. Needs adjustmenet if camera changes zoom

    
    if Square_root==True:
        data_May['Root_length']=np.sqrt(data_May['Root_length']/(538*imageArea))
        data_June['Root_length']=np.sqrt(data_June['Root_length']/(538*imageArea))
        data_July['Root_length']=np.sqrt(data_July['Root_length']/(538*imageArea))     
    else:
        data_May['Root_length']=data_May['Root_length']/(538*imageArea)
        data_June['Root_length']=data_June['Root_length']/(538*imageArea)
        data_July['Root_length']=data_July['Root_length']/(538*imageArea)

    #####Converting Tube Depth into soil depth based on which unit/bed of experiment. 
    #####Bed 1 and 2: Start_height =  570 mm, Angle = 23.5 degress. Start_Distance=420 mm
    #####Bed 3 and 4: Start_height =  330 mm, Angle = 15.8 degress. Start_Distance=420 mm

    data_May['soil_depth']=(((data_May['Depth']-420)*np.sin((23.5*(np.pi/180)))+570))/10. # in cm
    data_June['soil_depth']=(((data_June['Depth']-420)*np.sin((23.5*(np.pi/180)))+570))/10. # in cm
    data_July['soil_depth']=(((data_July['Depth']-420)*np.sin((23.5*(np.pi/180)))+570))/10. # in cm

    return [data_May,data_June,data_July]    
 
    
# Function to compute Root Lengths in different soil layers
def fun_RL_computation(RL_Raw_data,pre,n,depth_range): 
    """ Computation Root Lengths
    in Different Soil Layers """ 
    Tube=np.array([])
    RL_data = pd.DataFrame() 
    for i in set(RL_Raw_data.tube.sort_values()):
      RL_Sorted_by_Soil_Depth = RL_Raw_data[RL_Raw_data.tube==i].sort_values(by='soil_depth') # Arrange the data in increasing soil depth of i-th tube
      RL_Sorted_by_Soil_Depth= RL_Sorted_by_Soil_Depth.loc[:,['soil_depth','Root_length']] # Selecting only relevant columns
      RL_data=RL_data.append(pd.DataFrame(RL_Sorted_by_Soil_Depth.groupby(pd.cut(RL_Sorted_by_Soil_Depth['soil_depth'], depth_range))['Root_length'].mean()).transpose(),ignore_index = True) #  Function 'pd.cut' cuts RL into discrete intervals
      Tube=np.append(Tube,i)
    RL_data.fillna(RL_data.mean(),inplace=True)
    Intervals=RL_data.columns
    RL_data=RL_data.add_prefix(pre)
    RL_data.columns.name='Root Length :-' 
    RL_data.reset_index(drop=True,inplace=True)
    RL_data=pd.concat([RL_data,pd.DataFrame({'row':Tube})],axis=1) 
    return [RL_data,Intervals]


#Average root profile of all tubes in Bed 1 and Bed 2 for May, June and July 2018

def plot_RL(RL_May18,RL_June18,RL_July18,Intervals,n):
    #Mean root profiles of all the beds in May, June and July
    Mean_RL_May=RL_May18.iloc[:,:-1].mean(axis=0) # -1 due to last column is Tube number
    errMean_RL_May=RL_May18.iloc[:,:-1].std(axis=0) # -1 due to last column is Tube number

    Mean_RL_June=RL_June18.iloc[:,:-1].mean(axis=0)
    errMean_RL_June=RL_June18.iloc[:,:-1].std(axis=0)

    Mean_RL_July=RL_July18.iloc[:,:-1].mean(axis=0)
    errMean_RL_July=RL_July18.iloc[:,:-1].std(axis=0)

    Mean_RL_May.index=Intervals
    errMean_RL_May.index=Intervals
    Mean_RL_June.index=Intervals
    errMean_RL_June.index=Intervals
    Mean_RL_July.index=Intervals  
    errMean_RL_July.index=Intervals  
   # ax.bar(x_pos, CTEs,yerr=error,align='center',alpha=0.5,ecolor='black',capsize=10)
  #  Mean_RL=pd.DataFrame({'May':Mean_RL_May,'June':Mean_RL_June,'July':Mean_RL_July})
    MayMean_RL=pd.DataFrame({'May':Mean_RL_May,'stdErr':errMean_RL_May})
    #Mean_RL['stdErr']=3*Mean_RL['stdErr']
    #plt.errorbar(Mean_RL.index, 'May', yerr='stdErr', data=Mean_RL)
    ax1=MayMean_RL.plot(figsize=(8,4),yerr='stdErr',capsize=4)
    JuneMean_RL=pd.DataFrame({'June':Mean_RL_June,'stdErr':errMean_RL_June})
    #Mean_RL['stdErr']=3*Mean_RL['stdErr']
    #plt.errorbar(Mean_RL.index, 'May', yerr='stdErr', data=Mean_RL)
    JuneMean_RL.plot(ax=ax1,yerr='stdErr',capsize=6)
    JulyMean_RL=pd.DataFrame({'July':Mean_RL_July,'stdErr':errMean_RL_July})
    #Mean_RL['stdErr']=3*Mean_RL['stdErr']
    #plt.errorbar(Mean_RL.index, 'May', yerr='stdErr', data=Mean_RL)
    JulyMean_RL.plot(ax=ax1,yerr='stdErr',capsize=8)
    

    #plt.xticks(np.arange(n), Intervals,rotation=70,size=10) # use this for general
    #plt.xticks(np.arange(10), [str(119)+str(' - ')+str(129), [129, 140], [140, 150], [150, 160], [160, 170],
        #          [170, 180], [180, 190], [190, 200], [200, 210], [210, 220]],rotation=70,size=10) 
    plt.xticks(np.arange(10), [str(119)+str(' - ')+str(129), str(129)+str(' - ')+str(140), str(140)+str(' - ')+str(150), str(150)+str(' - ')+str(160), str(160)+str(' - ')+str(170),
                  str(170)+str(' - ')+str(180), str(180)+str(' - ')+str(190), str(190)+str(' - ')+str(200), str(200)+str(' - ')+str(210), str(210)+str(' - ')+str(220)],rotation=70,size=10) 

    plt.yticks(size=10)
    plt.legend(fontsize=10,loc="upper right")

    #plt.title('Average root profile of all tubes in Bed 1 and Bed 2 for May, June and July 2018')
    plt.ylabel('Sqrt_pRLD per image (sqrt($cm^{-1}$))') 
    
    plt.xlabel('Soil depth interval (cm)') 
    #plt.grid()
    plt.show()
    
    
def isotope_data_preprocess(isotope_data,year,side='Both'):
    
    if year==2018:
        
        isotope_data_tube18=isotope_data[(isotope_data['direction']=='S')&(isotope_data['bed']==1)| (isotope_data['direction']=='N')&(isotope_data['bed']==2)]
        isotope_data_tube18.reset_index(drop=True,inplace=True)
        isotope_data_tube18=isotope_data_tube18.loc[:,['x','row','ID','bed','δ15N (Air)','δ13C (VPDB)']]
        isotope_data_tube18.rename(columns={'δ15N (Air)':'Delta_15N','δ13C (VPDB)':'Delta_13C'}, inplace=True)
        isotope_data_tube18['Log_Delta_15N']=np.log(isotope_data_tube18['Delta_15N'])
        # Removing the missing values
        isotope_data_tube18.dropna(inplace=True)                                               
        return isotope_data_tube18
    
    elif year==2019:
        
        if side=='Both':
        
            idata=isotope_data[['x','bed','row','ID','position','d_13C','delta_15N']].copy()
            idata.drop([478,479,160],inplace=True)
            idatap_T=idata[(idata['position']=='South')&(idata['bed']==1)| (idata['position']=='North')&(idata['bed']==2)]
            idatap_T.reset_index(drop=True, inplace=True)
            idatap_NT=idata[(idata['position']=='North')&(idata['bed']==1)| (idata['position']=='South')&(idata['bed']==2)]
            idatap_NT.reset_index(drop=True, inplace=True)
            idatap_concat = pd.merge(idatap_T, idatap_NT,on=['bed','ID','row'])
            idatap_concat['Avg_Delta15N']=(idatap_concat.delta_15N_x+idatap_concat.delta_15N_y)/2
            idatap_concat['Avg_Delta13C']=(idatap_concat.d_13C_x+idatap_concat.d_13C_y)/2
            idatap_concat=idatap_concat[['x_x','bed','row','ID','Avg_Delta15N','Avg_Delta13C']].copy()
            idatap_concat.rename(columns={'x_x':'x','Avg_Delta15N':'Delta_15N','Avg_Delta13C':'Delta_13C'}, inplace=True)
            idatap_concat['Log_Delta_15N']=np.log(idatap_concat['Delta_15N'])
            idatap_concat.dropna(inplace=True)
            isotope_data_tube_19=idatap_concat
            return isotope_data_tube_19
        
        elif side=='Single':
            
            isotope_data_tube19=isotope_data[(isotope_data['position']=='South')&(isotope_data['bed']==1)| (isotope_data['position']=='North')&(isotope_data['bed']==2)]
            isotope_data_tube19.reset_index(drop=True,inplace=True)
            isotope_data_tube19=isotope_data_tube19.loc[:,['x','row','ID','bed','delta_15N','d_13C']]
            isotope_data_tube19.rename(columns={'delta_15N':'Delta_15N','d_13C':'Delta_13C'}, inplace=True)
            isotope_data_tube19['Log_Delta_15N']=np.log(isotope_data_tube19['Delta_15N'])
            # Removing the missing values
            isotope_data_tube19.dropna(inplace=True)                                               
            return isotope_data_tube19
    
        else:
            raise Exception('Wrong Side')
           
        
    
    else: 
        raise Exception('Wrong year')

        