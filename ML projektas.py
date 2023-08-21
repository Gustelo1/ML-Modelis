import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import joblib
from sklearn.metrics import mean_squared_error
#Nuskaitomi duomenys ir pakeičiama duomenų tvarka. Tvarka pakeičiama iš plačios į ilgą, kur 'ISO' ir 'WEO Subject Code'
# tampa identifikuojančiais kintamaisiais. Stulpelis pavadinimu 'year' tampa stulpelius, kuriame bus duomenų seto stulpelių
# pavadinimai, o value stulpelis turės visų metų duomenis.
df =  pd.read_excel(r'XLS FAILAS - Copy.xls' , engine = 'xlrd')
df = df.melt(
    id_vars=["ISO", "WEO Subject Code",],
    var_name="year",
    value_name="value", value_vars=[year for year in range(1980, 2026)])

#Pašalinamos Nan reikšmės value stulpelyje.
df = df.dropna(subset=["value"])


#Duomenų tvarka pakeičiama iš ilgos į plačią. Indeksais taps stulpeliai 'ISO' ir 'year', stulpeliais visi WEO
#Subject Code pavadinimai, o value taps reikšmėmis.
df = df.pivot_table(index=["ISO", "year"],
                      columns="WEO Subject Code",
                      values="value",
                      aggfunc='first',

)

#Grąžinimas senas indeksas
df.reset_index(inplace=True)


#X tampa features, y tampa targetu machine learningui.
X = df[['PCPIE' ,'GGSB' ,'LP', 'LE' ,'BCA']].dropna()
y = df[['NGDPDPC']].dropna()

#Randamos bendros eilutės tarp X ir y pagal indeksą. Taip yra sulyginami duomenų setai.
common_indices = X.index.intersection(y.index)
X = X.loc[common_indices]
y = y.loc[common_indices]
# X ir y duomenys pakeičiami į float tipą.
X = X.astype('float64')
y = y.astype('float64')

#Treniravimo duomenys splitinami, kad mokymas būtų teisingas
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

#Sukuriamas modelis su XGBRegressor. n_estimators ir learning_rate parinkti spėliojimo būdu. Didinant n_estimators skaičių mae mažėja, tačiau mažėjimas
#beveik nepastebimas, tad norint sutaupyti mokymui reikalingą laiką parinktas 500 skaičius.
my_model = XGBRegressor(n_estimators=500)

#Modelis fitinamas su mokymo duomenimis.
#Duodami validation duomenys, kad galima būtų sekti geriausius rezultatus. Taip nustatomas geriausias n_estimators skaičius.
my_model.fit(train_X, train_y, early_stopping_rounds=15, eval_set=[(val_X, val_y)], verbose=False)

# Get the best number of estimators from the training process
best_n_estimators = my_model.best_iteration

print("Best number of estimators:", best_n_estimators)
#Modeliui suteikiama mokymosi informacija.

#Atliekami modelio spėjimai.
preds_train = my_model.predict(train_X)
preds_val = my_model.predict(val_X)
#Apskaičiuojamas 'Mean Squared Error', kad žinotume kokio tikslumo yra modelis treniravimo informacijai.
mse_train = mean_squared_error(train_y, preds_train)

#Apskaičiuojamas 'Mean Squared Error', kad žinotume kokio tikslumo yra modelis testavimo informacijai.
mse_val = mean_squared_error(val_y, preds_val)
print("MSE on the training set:", mse_train)
print("MSE on the testing set:", mse_val)
#Naudojant joblib biblioteką modelis yra išsaugojamas 'trained_model.joblib' faile.
joblib.dump(my_model, 'trained_model.joblib')
