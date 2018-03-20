import lightgbm as lgb
import xgboost as xgb
import pandas as pd 
from sklearn.metrics import roc_auc_score as auc
path='/root/LightGBM/examples/regression/'
tr=pd.read_csv(path+'regression.train',header=None,sep='\t')
te=pd.read_csv(path+'regression.test',header=None,sep='\t')
ytr,xtr=tr[0].values,tr.drop(0,axis=1).values
yte,xte=te[0].values,te.drop(0,axis=1).values
lgb_tr,lgb_val=lgb.Dataset(xtr,ytr),lgb.Dataset(xte,yte)
param_lgb = {
'task': 'train',
'boosting_type': 'gbdt',
'objective': 'binary',
'metric': {'l2', 'auc'},
'num_leaves': 31,
'learning_rate': 0.05,
'feature_fraction': 0.9,
'bagging_fraction': 0.8,
'bagging_freq': 5,
'verbose': 0
}
gbm=lgb.train(param_lgb,lgb_tr,num_boost_round=50)
yp_lgb=gbm.predict(xte)

xlf = xgb.XGBRegressor(max_depth=10, 
                        learning_rate=0.05, 
                        n_estimators=50, 
                        silent=True, 
                        objective='binary:logistic', 
                        nthread=-1, 
                        gamma=0,
                        min_child_weight=1, 
                        max_delta_step=0, 
                        subsample=0.85, 
                        colsample_bytree=0.7, 
                        colsample_bylevel=1, 
                        reg_alpha=0, 
                        reg_lambda=1, 
                        scale_pos_weight=1, 
                        seed=1440, 
                        missing=None)

xlf.fit(xtr,ytr,eval_metric='logloss')
yp_xgb=xlf.predict(xte)

print 'auc(LGB)',auc(yte,yp_lgb)
print 'auc(XGB)',auc(yte,yp_xgb)

