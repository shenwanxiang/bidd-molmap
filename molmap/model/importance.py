from sklearn.metrics import mean_squared_error
from scipy.stats.stats import pearsonr
from copy import copy




def ForwardPropFeatureImp(model, X_true, Y_true, mp):
    '''
    Leave one feature out  importance
    '''
    try:
        df_grid = mp.df_grid
    except:
        H_grid = mp.plot_grid()
        df_grid = mp.df_grid
        
    df_grid = df_grid.sort_values(['y', 'x']).reset_index(drop=True)
    Y = model.predict(X_true)
    mse = mean_squared_error(Y_true, Y)
    N, W, H, C = X_true.shape
    results = []
    for i in tqdm(range(len(df_grid)), ascii= True):
        ts = df_grid.iloc[i]
        y = ts.y
        x = ts.x
        X1 = copy(X_true)
        X1[:, y, x,:] = np.zeros(X1[:, y, x,:].shape)
        Y1 = model.predict(X1)
        mse_mutaion = mean_squared_error(Y_true, Y1)
        res = mse_mutaion - mse  # if res > 0, important, othervise, not important
        results.append(res)
            
    S = pd.Series(results, name = 'importance')
    df = df_grid.join(S)
    return df