
### **The Decision Tree Model showed the next data:**  
**A code snippet of the model instantiation:**  
- Split the data into training and testing sets  

X_train, X_test, y_train, y_test = train_test_split(  
    X, y, test_size=0.22, random_state=42  
)  

- Create and train the Decision Tree model  

tr_regressor = DecisionTreeRegressor(random_state=42)  
tr_regressor.fit(X_train, y_train)  

**Training Set Performance:**  
*R² Score: 0.4610856873098689*  
*Mean Squared Error (MSE): 12220071488.587204*  
*Root Mean Squared Error (RMSE): 110544.43219170834*  
*Mean Absolute Error (MAE): 80249.05879262238*  

**Testing Set Performance:**  
*R² Score: 0.36041596837416723*  
*Mean Squared Error (MSE): 14018524590.981222*  
*Root Mean Squared Error (RMSE): 118399.85046857627*  
*Mean Absolute Error (MAE): 86860.86971330237*  

**The list of features:**  
- 'Type_of_Property'  
- 'Living_Area'  
- 'Region_Code'  

**Divide the dataset into training and testing sets**:  
This is a crucial step. I divided my data into two parts: one for training the model (X_train, y_train) and another for testing and evaluating the model's performance (X_test, y_test). I tried two variants: 0.22 (1/5) and 0.33 (1/3). **The best resulst was with the 20/80 splitting.**  

**Efficiency:** 1 sec.  

### **The Random Forest Model showed the next data:**  

- Parameters:  

self.rf_regressor = RandomForestRegressor(  
            n_estimators=100,  
            max_depth=10,  
            min_samples_split=20,  
            min_samples_leaf=10,  
            random_state=42,  
            n_jobs=-1,  
        )  

- Initialize Random Forest model  

rf_model = RandomForestModel(data_path)  

- Split data into training and testing sets  

X_train, X_test, y_train, y_test = train_test_split(  
    X, y, test_size=0.22, random_state=42  

**Training Set Performance:**  
*R² Score: 0.5557598892245861*  
*Mean Squared Error (MSE): 10073300678.682955*  
*Root Mean Squared Error (RMSE): 100365.83422003205*  
*Mean Absolute Error (MAE): 72321.4973339466*  

**Testing Set Performance:**  
*R² Score: 0.5050437364164431*  
*Mean Squared Error (MSE): 10848545631.866943*  
*Root Mean Squared Error (RMSE): 104156.3518555971*  
*Mean Absolute Error (MAE): 75687.4478682724*  

**Cross-Validated R² Scores:**  
[0.54500171 0.51135715 0.47766754 0.52613301 0.47682392 0.46105765 0.48708745 0.48663609 0.51339655 0.50352611]  

**Mean R² Score from Cross-Validation:** 0.49886871775284475  

**The list of features:**  
- "Type_of_Property"  
- "Number_of_Rooms"  
- "Living_Area"  
- "Fully_Equipped_Kitchen"  
- "Terrace"  
- "Garden"  
- "Surface_area_plot_of_land"  
- "Number_of_Facades"  
- "Region_Code"  
- "AS_NEW"  
- "GOOD"  
- "JUST_RENOVATED"  
- "TO_BE_DONE_UP"  
- "TO_RENOVATE"  
- "TO_RESTORE"  
- "APARTMENT"  
- "DUPLEX"  
- "HOUSE"  
- "PENTHOUSE"  
- "TOWN_HOUSE"  
- "VILLA"  

**Divide the dataset into training and testing sets**:  
The best resulst was with the 20/80 splitting.  

**Efficiency:** 6.8 sec.  

**Feature Importance:**  
Feature  Importance:  
2                 Living_Area    0.414775  
8                 Region_Code    0.271200  
3      Fully_Equipped_Kitchen    0.051657  
9                      AS_NEW    0.050627  
13                TO_RENOVATE    0.047689  
17                      HOUSE    0.036091  
6   Surface_area_plot_of_land    0.033003  
1             Number_of_Rooms    0.022142  
4                     Terrace    0.013456  
12              TO_BE_DONE_UP    0.012301  
0            Type_of_Property    0.009258  
7           Number_of_Facades    0.008031  
18                  PENTHOUSE    0.007085  
11             JUST_RENOVATED    0.005909  
10                       GOOD    0.005675  
15                  APARTMENT    0.004480  
5                      Garden    0.003483  
20                      VILLA    0.001739  
16                     DUPLEX    0.000734  
19                 TOWN_HOUSE    0.000663  
14                 TO_RESTORE    0.000000  

**A quick presentation of the final dataset**  
I wanted to change categorical data (municipality and province) to numeric (Brussels Capital - 1, Flanders - 2, Wallonia - 3).    
To do this, I add a new dataset (zip.csv), clean this dataset and merge it with the main dataset.   

As a result I received the dataset without NaN and categorical data with the next structure: **15.555,00 raws in 40 Columns**. It happened after merging dummies data from 'Subtype_of_Property' (17 columns) and 'State_of_the_Building' (5 columns).  

My target was a Price and others variables were as a features (39 features were tested in different ways and chosen the best four for modelling).  
