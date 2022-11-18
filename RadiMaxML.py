# Modelling using Random Forest 
import scipy
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold,GridSearchCV
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy import optimize
seed = np.random.seed(22)
from sklearn.metrics import mean_squared_error

def ML_NCV_OLD(df):
            #for storing model's scores
        ML_df=pd.DataFrame({'Lin':[0,0,0],'KNN':[0,0,0],'RF':[0,0,0],'GB':[0,0,0]}, index=['Delta_15N','Log_Delta_15N','Delta_13C']) 
        #ML_df=pd.DataFrame({'Lin':[0,0,0],'KNN':[0,0,0]}, index=['Delta_15N','Log_Delta_15N','Delta_13C']) 
        #for storing p-value of model's score
        # ML_df_pval ue=pd.DataFrame({'Lin':[0,0,0],'RF':[0,0,0],'GB':[0,0,0]}, index=['Delta_15N','Log_Delta_15N','Delta_13C'])
        ML_df_pvalue = ML_df.copy()
        #Nested Cross-validation for finding optimal hyper_params
        Lin_models_param_grid = [{}]
        knn_grid = [{'n_neighbors': [5,10,20,30]}]
        RF_models_param_grid = [ 
                          { #corresponding to RandomForestRegressor
                                  'max_depth': [5,10,20,None], #If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
                                  'n_estimators': [100,500],
                                   'max_features':[5,10,'auto','log2',None] #If None, then max_features=n_features.
                                  }]   
        RF_models_param_grid = [ 
                          { #corresponding to RandomForestRegressor
                                  'max_depth': [5,10], #If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
                                  'n_estimators': [500],
                                   'max_features':[5,10,'auto'] #If None, then max_features=n_features.
                                  }]   


        GB_models_param_grid ={'n_estimators': [100,200,300],
                'max_depth': [3,5, None],
                'min_samples_split': [3,5], # default=2, The minimum number of samples required to split an internal node:
                'learning_rate': [0.01,.001,.0001]}  

        Xall=df.iloc[:,0:-3].to_numpy()

        if False: # Do PCA
            print(Xall.shape)
            pca = PCA(n_components=10)
            Xall = pca.fit_transform(Xall)

        if True: # Normalize to zero mean, std one
            means = np.mean(Xall, 0)
            Xall += -np.mean(Xall, 0)
            Xall *= 1 / np.std(Xall, 0)

        #print(Xall.shape)
        #print(ML_df.index)

        # for loop : To get all the 6 results in one go
        for i in ML_df.index: #ML_df.index: Index(['Delta_15N', 'Log_Delta_15N', 'Delta_13C'], dtype='object')
            for j in ML_df.columns: #ML_df.columns: Index(['RF', 'GB'], dtype='object')
                print(' ')
                print(i,j)
                y=df.loc[:,i]
                y=y.to_numpy()
                X=Xall.copy()

                if j=='Lin':
                    model = LinearRegression()
                    grid = Lin_models_param_grid
                elif j=='RF':
                    model=RandomForestRegressor()
                    grid=RF_models_param_grid
                elif j=='KNN':
                    model = KNeighborsRegressor()
                    grid = knn_grid
                else:
                    model=GradientBoostingRegressor()
                    grid=GB_models_param_grid
               

                Pred_all=np.zeros(len(y))
                y_all=np.zeros(len(y)) 
                cv_outer = KFold(n_splits=5, shuffle=True, random_state=123) 
                outer_results = list()
                fold = 0
                for train_ix, test_ix in cv_outer.split(X):
                    # split data
                    X_train, X_test = X[train_ix, :], X[test_ix, :]
                    y_train, y_test = y[train_ix], y[test_ix]
                    # configure the cross-validation procedure       
                    search = GridSearchCV(model, grid, scoring='r2', cv=4, refit=True, n_jobs=-1, return_train_score=False)         
                    # execute search
                    result = search.fit(X_train, y_train)
                    # get the best performing model fit on the whole training set
                    # trainResults = result.cv_results_
                    # print(trainResults)
                    best_model = result.best_estimator_
                   
                    # evaluate model on the hold out dataset
                    Pred_all[test_ix]=best_model.predict(X_test) #copying results of predict(X_test) into Pred_all based on index of X_test i.e. test_ix
                    y_all[test_ix]=y[test_ix]
                    trainR, trainP = scipy.stats.pearsonr(y[train_ix],best_model.predict(X_train))
                    testR,  testP  = scipy.stats.pearsonr(y[test_ix ],best_model.predict(X_test ))
                    fold += 1
                    print('Fold %d, R train %.2f test %.2f' % (fold,trainR,testR))
                rp=scipy.stats.pearsonr(Pred_all, y_all)# Correlation on entire data set
                print('Full test r %.2f' % (rp[0]))
                ML_df.loc[[i],[j]]=rp[0]
                ML_df_pvalue.loc[[i],[j]]=rp[1]
                
        return [ML_df, ML_df_pvalue]
    
    
def ML_NCV(df,ML_grid,ML_model='RF'):
        #for storing model's scores
        #ML_df=pd.DataFrame({'Lin':[0,0,0],'KNN':[0,0,0],'RF':[0,0,0],'GB':[0,0,0]}, index =                                 ['Delta_15N','Log_Delta_15N','Delta_13C'])

        ML_df=pd.DataFrame({ML_model:[0,0,0]}, index=['Delta_15N','Log_Delta_15N','Delta_13C']) 
        
        ML_Pred=pd.DataFrame() 
        base=pd.DataFrame()

        #for storing p-value of model's score
        # ML_df_pval ue=pd.DataFrame({'Lin':[0,0,0],'RF':[0,0,0],'GB':[0,0,0]}, index=['Delta_15N','Log_Delta_15N','Delta_13C'])
        ML_df_pvalue = ML_df.copy()
        ML_df_TrainRMSE = ML_df.copy()
        ML_df_TestRMSE = ML_df.copy()
        #Nested Cross-validation for finding optimal hyper_params
        Lin_models_param_grid = [{}]
        knn_grid = [{'n_neighbors': [5,10,20,30]}]


        Xall=df.iloc[:,0:-5].to_numpy() # as x and bed are added in the end

        if False: # Do PCA
            print(Xall.shape)
            pca = PCA(n_components=10)
            Xall = pca.fit_transform(Xall)

        #if True: # Normalize to zero mean, std one
         #   print('Normalize X')
          #  means = np.mean(Xall, 0)
          #  Xall += -np.mean(Xall, 0)
           # Xall *= 1 / np.std(Xall, 0)
#
        #print(Xall.shape)
        #print(ML_df.index)

        # for loop : To get all the 6 results in one go
        FIMM=pd.DataFrame()
        for i in ML_df.index: #ML_df.index: Index(['Delta_15N', 'Log_Delta_15N', 'Delta_13C'], dtype='object')
            for j in ML_df.columns: #ML_df.columns: Index(['RF', 'GB'], dtype='object')
                print(' ')
                print(i,j)
                y=df.loc[:,i]
                y=y.to_numpy()
                X=Xall.copy()
                x_cord=df['x'].to_numpy()
               
                if j=='Lin':
                    model = LinearRegression()
                    grid = ML_grid
                elif j=='RF':
                    model=RandomForestRegressor()
                    grid=ML_grid
                elif j=='KNN':
                    model = KNeighborsRegressor()
                    grid = ML_grid
                else:
                    model=GradientBoostingRegressor()
                    grid=ML_grid
                FI=pd.DataFrame()    
                Pred_all=np.zeros(len(y))
                y_all=np.zeros(len(y))
                x_cord_all=np.zeros(len(y)) 
                cv_outer = KFold(n_splits=5, shuffle=True, random_state=123) 
                outer_results = list()
                fold = 0
                for train_ix, test_ix in cv_outer.split(X):
                    # split data
                    X_train, X_test = X[train_ix, :], X[test_ix, :]
                    y_train, y_test = y[train_ix], y[test_ix]
                    # configure the cross-validation procedure       
                    search = GridSearchCV(model, grid, scoring='r2', cv=4, refit=True, n_jobs=-1, return_train_score=False)         
                    # execute search
                    result = search.fit(X_train, y_train)
                    # get the best performing model fit on the whole training set
                    # trainResults = result.cv_results_
                    # print(trainResults)
                    best_model = result.best_estimator_
                    print(search.best_params_)               
                    # evaluate model on the hold out dataset
                    Pred_all[test_ix]=best_model.predict(X_test) #copying results of predict(X_test) into Pred_all based on index of X_test i.e. test_ix
                    y_all[test_ix]=y[test_ix]
                    x_cord_all[test_ix]= x_cord[test_ix]
                    trainR, trainP = scipy.stats.pearsonr(y[train_ix],best_model.predict(X_train))
                    testR,  testP  = scipy.stats.pearsonr(y[test_ix ],best_model.predict(X_test )) 
                    FI=pd.concat([FI,pd.DataFrame({j+'FI'+str(fold):best_model.feature_importances_})],axis=1)


                    
                    print('Fold %d, R train %.2f test %.2f' % (fold,trainR,testR))
                    name='Pred_'+i+'_'+str(fold)
                    #print(name)
                    df[name]=best_model.predict(X)
                    fold += 1
                rp=scipy.stats.pearsonr(Pred_all, y_all)# Correlation on entire data set
                TestRMSE=mean_squared_error(Pred_all,  y_all, squared=True)
                base=pd.concat([base,pd.DataFrame({'x':x_cord_all, 'RFP_'+i:Pred_all})],axis=1)

                print('Full test r %.2f' % (rp[0]))
                ML_df.loc[[i],[j]]=rp[0]
                ML_df_pvalue.loc[[i],[j]]=rp[1]
                ML_df_TestRMSE.loc[[i],[j]]=TestRMSE
                FIM=FI.mean(axis=1)
                FIMM=pd.concat([FIMM,pd.DataFrame({i:FIM})],axis=1)
        df.to_csv('df_ML_pred.csv')# Need for Mediation Analysius
        base.to_csv('xMlP.csv')# Need for Mediation Analysius    
        return [ML_df,FIMM,ML_df_pvalue,ML_df_TestRMSE]


    
    

    

def piecewise_linear(x, x0, a, b, c):
        y = a*np.abs(x-x0) + b*x + c
        return y
def piecewise_Lin_fit(x,y,x0):
    p , e = optimize.curve_fit(piecewise_linear, x, y,p0=[x0, 0, 0, 1],maxfev = 10000)
    return piecewise_linear(x, *p)
def fun_Spatial_Corrected(x,y,x0):
  #x=x.to_numpy().reshape(-1,1) # for 2D
  y_pred= piecewise_Lin_fit(x.values,y.values,x0)
  y_corrected=y-y_pred
  return y_corrected

def fun_Spatial_Corrected_Linear(x,y):
  x=x.to_numpy().reshape(-1,1) # for 2D
  regressor = LinearRegression() 
  regressor.fit(x, y)
  y_pred= regressor.predict(x)
  y_corrected=y-y_pred
  return y_corrected

def SpatialCorrection(df,type_spatial_normaization):
    
    # Partitions for spatial Anlysis in Bed1 and Bed2 separately
    dfb1=df[df.bed==1]
    dfb2=df[df.bed==2]
    cols = list(dfb1.columns.values) #Make a list of all of the columns in the df
    cols.pop(cols.index('bed')) #Remove bed from list
    cols.pop(cols.index('x')) #Remove x from list
    dfb1 = dfb1[cols+['bed','x']]
    dfb2 = dfb2[cols+['bed','x']] #Create new dataframe with columns in the order you want
    
    if type_spatial_normaization=='L':    
        
        for i in set(dfb1.columns.values).difference(set(['bed','x'])):
          Normalize=fun_Spatial_Corrected_Linear(dfb1['x'],dfb1[i])
          dfb1[i]= Normalize.values
        for i in set(dfb2.columns.values).difference(set(['bed','x'])):
          Normalize=fun_Spatial_Corrected_Linear(dfb2['x'],dfb2[i])
          dfb2[i]= Normalize.values
        dfs=pd.concat([dfb1,dfb2])
        #df_final=dfs.loc[:, ~dfs.columns.isin(['x', 'bed'])]
        print('\n Linear correction: Done')
        return  dfs


    else:
        
        for i in set(dfb1.columns.values).difference(set(['bed','x'])):
          Normalize=fun_Spatial_Corrected(dfb1['x'],dfb1[i],x0=1200)
          dfb1[i]= Normalize.values
        for i in set(dfb2.columns.values).difference(set(['bed','x'])):
          Normalize=fun_Spatial_Corrected(dfb2['x'],dfb2[i],x0=2200)
          dfb2[i]= Normalize.values
        dfs=pd.concat([dfb1,dfb2])
        #df_final=dfs.loc[:, ~dfs.columns.isin(['x', 'bed'])]
        print('\n Piecewise Linear correction: Done')
        return  dfs

    
def Feature_Importances(df,RF_models_param_grid):
                #for storing model's scores
            #ML_df=pd.DataFrame({'Lin':[0,0,0],'KNN':[0,0,0],'RF':[0,0,0],'GB':[0,0,0]}, index=['Delta_15N','Log_Delta_15N','Delta_13C']) 
            ML_df=pd.DataFrame({'RF':[0,0,0]}, index=['Delta_15N','Log_Delta_15N','Delta_13C']) 
            #for storing p-value of model's score
            # ML_df_pval ue=pd.DataFrame({'Lin':[0,0,0],'RF':[0,0,0],'GB':[0,0,0]}, index=['Delta_15N','Log_Delta_15N','Delta_13C'])
            ML_df_pvalue = ML_df.copy()
            #Nested Cross-validation for finding optimal hyper_params
            Lin_models_param_grid = [{}]
            
            
            
            '''knn_grid = [{'n_neighbors': [5,10,20,30]}]

            GB_models_param_grid ={'n_estimators': [100,200,300],
                    'max_depth': [3,5, None],
                    'min_samples_split': [3,5], # default=2, The minimum number of samples required to split an internal node:
                    'learning_rate': [0.01,.001,.0001]}  
            RF_models_param_grid = [ 
                          { #corresponding to RandomForestRegressor
                                  'max_depth': [2,3,5,10,15], #If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
                                  'n_estimators': [50,100,200,300,500,1000],
                                   'max_features':[3,5,10,20, None] #If None, then max_features=n_features.
                                  }]   '''


            # Need for Mediation Analysius

            Xall=df.iloc[:,0:-5].to_numpy() #as x and bed are added in the end

            #print(Xall.shape)
            #print(ML_df.index)
            
            # for loop : To get all the 6 results in one go
            FIMM=pd.DataFrame()
            for i in ML_df.index: #ML_df.index: Index(['Delta_15N', 'Log_Delta_15N', 'Delta_13C'], dtype='object')
                
                for j in ML_df.columns: #ML_df.columns: Index(['RF', 'GB'], dtype='object')
                    print(' ')
                    print(i,j)
                    y=df.loc[:,i]
                    y=y.to_numpy()
                    X=Xall.copy()

                    model=RandomForestRegressor()
                    grid=RF_models_param_grid
                    
                        
                    FI=pd.DataFrame()
    

                    Pred_all=np.zeros(len(y))
                    y_all=np.zeros(len(y)) 
                    cv_outer = KFold(n_splits=5, shuffle=True, random_state=123) 
                    outer_results = list()
                    fold = 0
                    for train_ix, test_ix in cv_outer.split(X):
                        # split data
                        X_train, X_test = X[train_ix, :],X[test_ix, :]
                        y_train, y_test = y[train_ix], y[test_ix]
                        # configure the cross-validation procedure       
                        search = GridSearchCV(model, grid, scoring='r2', cv=4, refit=True, n_jobs=-1, return_train_score=False)         
                        # execute search
                        result = search.fit(X_train, y_train)
                        # get the best performing model fit on the whole training set
                        # trainResults = result.cv_results_
                        # print(trainResults)
                        best_model = result.best_estimator_
                        FI=pd.concat([FI,pd.DataFrame({j+'FI'+str(fold):best_model.feature_importances_})],axis=1)

                        # evaluate model on the hold out dataset
                        Pred_all[test_ix]=best_model.predict(X_test) #copying results of predict(X_test) into Pred_all based on index of X_test i.e. test_ix
                        y_all[test_ix]=y[test_ix]
                        trainR, trainP = scipy.stats.pearsonr(y[train_ix],best_model.predict(X_train))
                        testR,  testP  = scipy.stats.pearsonr(y[test_ix ],best_model.predict(X_test ))
                        fold += 1
                        print('Fold %d, R train %.2f test %.2f' % (fold,trainR,testR))
                    rp=scipy.stats.pearsonr(Pred_all, y_all)# Correlation on entire data set
                    print('Full test r %.2f' % (rp[0]))
                    ML_df.loc[[i],[j]]=rp[0]
                    ML_df_pvalue.loc[[i],[j]]=rp[1]
                    FIM=FI.mean(axis=1)
                    print(FIM)
                    print(i)    
                    FIMM=pd.concat([FIMM,pd.DataFrame({i:FIM})],axis=1)
                    
            
            return [FIMM,ML_df]

               
def Feature_Importances_OLD(df):
                #for storing model's scores
            #ML_df=pd.DataFrame({'Lin':[0,0,0],'KNN':[0,0,0],'RF':[0,0,0],'GB':[0,0,0]}, index=['Delta_15N','Log_Delta_15N','Delta_13C']) 
            ML_df=pd.DataFrame({'RF':[0,0,0]}, index=['Delta_15N','Log_Delta_15N','Delta_13C']) 
            #for storing p-value of model's score
            # ML_df_pval ue=pd.DataFrame({'Lin':[0,0,0],'RF':[0,0,0],'GB':[0,0,0]}, index=['Delta_15N','Log_Delta_15N','Delta_13C'])
            ML_df_pvalue = ML_df.copy()
            #Nested Cross-validation for finding optimal hyper_params
            Lin_models_param_grid = [{}]
            knn_grid = [{'n_neighbors': [5,10,20,30]}]
            RF_models_param_grid = [ 
                              { #corresponding to RandomForestRegressor
                                      'max_depth': [5,10], #If None, then nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_split samples.
                                      'n_estimators': [500],
                                       'max_features':[5,10,'auto'] #If None, then max_features=n_features.
                                      }]   


            GB_models_param_grid ={'n_estimators': [100,200,300],
                    'max_depth': [3,5, None],
                    'min_samples_split': [3,5], # default=2, The minimum number of samples required to split an internal node:
                    'learning_rate': [0.01,.001,.0001]}  

            Xall=df.iloc[:,0:-3].to_numpy()

            #print(Xall.shape)
            #print(ML_df.index)
            
            # for loop : To get all the 6 results in one go
            FIMM=pd.DataFrame()
            for i in ML_df.index: #ML_df.index: Index(['Delta_15N', 'Log_Delta_15N', 'Delta_13C'], dtype='object')
                
                for j in ML_df.columns: #ML_df.columns: Index(['RF', 'GB'], dtype='object')
                    print(' ')
                    print(i,j)
                    y=df.loc[:,i]
                    y=y.to_numpy()
                    X=Xall.copy()

                    model=RandomForestRegressor()
                    grid=RF_models_param_grid                     
                    FI=pd.DataFrame()
    

                    Pred_all=np.zeros(len(y))
                    y_all=np.zeros(len(y)) 
                    cv_outer = KFold(n_splits=5, shuffle=True, random_state=123) 
                    outer_results = list()
                    fold = 0
                    for train_ix, test_ix in cv_outer.split(X):
                        # split data
                        X_train, X_test = X[train_ix, :],X[test_ix, :]
                        y_train, y_test = y[train_ix], y[test_ix]
                        # configure the cross-validation procedure       
                        search = GridSearchCV(model, grid, scoring='r2', cv=4, refit=True, n_jobs=-1, return_train_score=False)         
                        # execute search
                        result = search.fit(X_train, y_train)
                        # get the best performing model fit on the whole training set
                        # trainResults = result.cv_results_
                        # print(trainResults)
                        best_model = result.best_estimator_
                        FI=pd.concat([FI,pd.DataFrame({j+'FI'+str(fold):best_model.feature_importances_})],axis=1)

                        # evaluate model on the hold out dataset
                        Pred_all[test_ix]=best_model.predict(X_test) #copying results of predict(X_test) into Pred_all based on index of X_test i.e. test_ix
                        y_all[test_ix]=y[test_ix]
                        trainR, trainP = scipy.stats.pearsonr(y[train_ix],best_model.predict(X_train))
                        testR,  testP  = scipy.stats.pearsonr(y[test_ix ],best_model.predict(X_test ))
                        fold += 1
                        print('Fold %d, R train %.2f test %.2f' % (fold,trainR,testR))
                    rp=scipy.stats.pearsonr(Pred_all, y_all)# Correlation on entire data set
                    print('Full test r %.2f' % (rp[0]))
                    ML_df.loc[[i],[j]]=rp[0]
                    ML_df_pvalue.loc[[i],[j]]=rp[1]
                    FIM=FI.mean(axis=1)
                    print(FIM)
                    print(i)    
                    FIMM=pd.concat([FIMM,pd.DataFrame({i:FIM})],axis=1)
                    
            
            return [FIMM,ML_df]



        
