import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

st.set_page_config(page_title="Bank Customer Churn Prediction", layout="wide")
st.title("Bank Customer Churn Prediction Dashboard")
st.caption("Predictive Modeling and Risk Scoring | European Bank Dataset")

@st.cache_data
def load_data(): return pd.read_csv("European_Bank.csv")

def engineer(df):
    out=df.copy()
    out["BalanceToSalaryRatio"]=out["Balance"]/(out["EstimatedSalary"]+1)
    out["ProductDensity"]=out["NumOfProducts"]/(out["Tenure"]+1)
    out["EngagementProductInteraction"]=out["IsActiveMember"]*out["NumOfProducts"]
    out["AgeTenureInteraction"]=out["Age"]*out["Tenure"]
    return out

@st.cache_resource
def train_model(df):
    d=engineer(df); X=d.drop(columns=["Exited","CustomerId","Surname","Year"]); y=d["Exited"]
    cat=["Geography","Gender"]; num=[c for c in X.columns if c not in cat]
    prep=ColumnTransformer([
        ("num",Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]),num),
        ("cat",Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]),cat)])
    model=XGBClassifier(n_estimators=300,max_depth=4,learning_rate=0.05,subsample=0.9,colsample_bytree=0.9,
                        objective="binary:logistic",eval_metric="logloss",random_state=42,n_jobs=4)
    pipe=Pipeline([("prep",prep),("model",model)])
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.2,stratify=y,random_state=42)
    pipe.fit(Xtr,ytr); p=pipe.predict_proba(Xte)[:,1]; pred=(p>=0.5).astype(int)
    m={"Accuracy":accuracy_score(yte,pred),"Precision":precision_score(yte,pred),"Recall":recall_score(yte,pred),
       "F1":f1_score(yte,pred),"ROC-AUC":roc_auc_score(yte,p)}
    return pipe,m

df=load_data(); model,metrics=train_model(df)
scored=engineer(df.copy()); Xall=scored.drop(columns=["Exited","CustomerId","Surname","Year"])
scored["ChurnProbability"]=model.predict_proba(Xall)[:,1]
scored["RiskBand"]=pd.cut(scored["ChurnProbability"],[-0.01,0.30,0.60,1.01],labels=["Low","Medium","High"])

c1,c2,c3,c4=st.columns(4)
c1.metric("Total Customers",f"{len(df):,}")
c2.metric("Observed Churn Rate",f"{df.Exited.mean()*100:.2f}%")
c3.metric("XGBoost ROC-AUC",f"{metrics['ROC-AUC']:.3f}")
c4.metric("High-Risk Customers",f"{(scored.RiskBand=='High').sum():,}")
st.subheader("Risk Distribution")
st.bar_chart(scored["RiskBand"].value_counts().reindex(["Low","Medium","High"]))

st.subheader("Customer Churn Risk Calculator")
l,r=st.columns(2)
with l:
    geography=st.selectbox("Geography",["France","Spain","Germany"]); gender=st.selectbox("Gender",["Male","Female"])
    age=st.number_input("Age",18,100,40); tenure=st.number_input("Tenure",0,10,5)
    balance=st.number_input("Balance",0.0,300000.0,75000.0)
with r:
    products=st.number_input("Number of Products",1,4,1)
    card=st.selectbox("Has Credit Card",[0,1],format_func=lambda x:"Yes" if x else "No")
    active=st.selectbox("Active Member",[0,1],format_func=lambda x:"Yes" if x else "No")
    salary=st.number_input("Estimated Salary",0.0,250000.0,100000.0)

customer=pd.DataFrame([{"CreditScore":650,"Geography":geography,"Gender":gender,"Age":age,"Tenure":tenure,
"Balance":balance,"NumOfProducts":products,"HasCrCard":card,"IsActiveMember":active,"EstimatedSalary":salary}])
customer=engineer(customer); prob=float(model.predict_proba(customer)[:,1][0])
risk="Low" if prob<0.30 else ("Medium" if prob<0.60 else "High")
st.metric("Predicted Churn Probability",f"{prob*100:.2f}%"); st.info(f"Risk Band: {risk}")

st.subheader("What-if Scenario")
new_active=st.slider("Scenario: Active Member",0,1,int(active)); new_products=st.slider("Scenario: Number of Products",1,4,int(products))
scenario=customer.copy(); scenario["IsActiveMember"]=new_active; scenario["NumOfProducts"]=new_products
scenario["EngagementProductInteraction"]=new_active*new_products; scenario["ProductDensity"]=new_products/(tenure+1)
scenario_prob=float(model.predict_proba(scenario)[:,1][0])
st.write(f"Scenario churn probability: **{scenario_prob*100:.2f}%**")
st.caption("Scenario outputs show model sensitivity and are not causal intervention estimates.")
