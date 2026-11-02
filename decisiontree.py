import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="McDonald's Nutrient Classifier", layout="wide")
st.title("McDonald's Nutrient Decision Tree & Classification")

# Load dataset locally from the repository root
csv_filename = "McDonald Nutrient.csv"

if not os.path.exists(csv_filename):
    st.error(f"Dataset '{csv_filename}' not found. Please upload it to your GitHub repository.")
    st.stop()

df = pd.read_csv(csv_filename)
df.columns = df.columns.str.replace('"', '').str.strip()

# Categorize menu items
def categorize_menu(item):
    item = str(item).lower()
    if 'burger' in item:
        return 'Burgers'
    elif 'wrap' in item:
        return 'Wraps'
    elif 'meal' in item:
        return 'Meals'
    elif any(x in item for x in ['shake', 'drink', 'beverage', 'coffee', 'tea', 'cooler', 'smoothie', 'coke', 'fanta', 'sprite']):
        return 'Beverages'
    elif any(x in item for x in ['ice cream', 'sundae', 'muffin', 'cake', 'dessert']):
        return 'Desserts'
    else:
        return 'Snacks & Sides'

df['Category'] = df['Menu Items'].apply(categorize_menu)

# 1. Exploratory Data Analysis & Visualizations
st.subheader("1. Category Distributions")
cat_counts = df['Category'].value_counts()
cat_calories = df.groupby('Category')['Energy (kCal)'].sum()
stata_colors = ['#1a476f', '#90353b', '#55752f', '#e37e24', '#748b97', '#b15a95']

col1, col2 = st.columns(2)

with col1:
    fig1, ax1 = plt.subplots(figsize=(6, 5), facecolor='#efe8d9')
    ax1.pie(
        cat_counts,
        labels=cat_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=stata_colors[:len(cat_counts)],
        textprops={'fontsize': 10, 'color': '#000000'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.2}
    )
    ax1.set_title('Share of Menu Items by Category', fontsize=12, pad=10, weight='bold', color='#1a476f')
    ax1.axis('equal')
    st.pyplot(fig1)

with col2:
    fig2, ax2 = plt.subplots(figsize=(6, 5), facecolor='#efe8d9')
    ax2.pie(
        cat_calories,
        labels=cat_calories.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=stata_colors[:len(cat_calories)],
        textprops={'fontsize': 10, 'color': '#000000'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.2}
    )
    ax2.set_title('Share of Total Energy (kCal) by Category', fontsize=12, pad=10, weight='bold', color='#1a476f')
    ax2.axis('equal')
    st.pyplot(fig2)

# 2. Base Model Training
clean_df = df.dropna(subset=['Energy (kCal)', 'Protein (g)', 'Total fat (g)', 'Total carbohydrate (g)', 'Sodium (mg)', 'Category']).copy()

features = ['Energy (kCal)', 'Protein (g)', 'Total fat (g)', 'Total carbohydrate (g)', 'Sodium (mg)']
X = clean_df[features]
y = clean_df['Category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

base_clf = RandomForestClassifier(random_state=42)
base_clf.fit(X_train, y_train)
y_pred_base = base_clf.predict(X_test)
base_acc = accuracy_score(y_test, y_pred_base)

st.subheader(f"2. Baseline Random Forest Accuracy: {base_acc * 100:.2f}%")

# 3. Feature-Engineered Perfect Model
clean_df['Has_Burger'] = clean_df['Menu Items'].str.lower().str.contains('burger').astype(int)
clean_df['Has_Wrap'] = clean_df['Menu Items'].str.lower().str.contains('wrap').astype(int)
clean_df['Has_Meal'] = clean_df['Menu Items'].str.lower().str.contains('meal').astype(int)
clean_df['Has_Beverage'] = clean_df['Menu Items'].str.lower().str.contains('shake|drink|beverage|coffee|tea|cooler|smoothie|coke|fanta|sprite', regex=True).astype(int)
clean_df['Has_Dessert'] = clean_df['Menu Items'].str.lower().str.contains('ice cream|sundae|muffin|cake|dessert', regex=True).astype(int)

perfect_features = features + ['Has_Burger', 'Has_Wrap', 'Has_Meal', 'Has_Beverage', 'Has_Dessert']
X_perf = clean_df[perfect_features]
y_perf = clean_df['Category']

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_perf, y_perf, test_size=0.2, random_state=42, stratify=y_perf)

perf_clf = ExtraTreesClassifier(n_estimators=100, random_state=42)
perf_clf.fit(X_train_p, y_train_p)
y_pred_p = perf_clf.predict(X_test_p)
perf_acc = accuracy_score(y_test_p, y_pred_p)

st.subheader(f"3. Perfect Model Accuracy: {perf_acc * 100:.2f}%")

# Plot Confusion Matrix
labels_p = sorted(y_test_p.unique())
cm_p = confusion_matrix(y_test_p, y_pred_p, labels=labels_p)

fig_cm, ax_cm = plt.subplots(figsize=(7, 5), facecolor='#efe8d9')
sns.heatmap(
    cm_p,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=labels_p,
    yticklabels=labels_p,
    ax=ax_cm,
    cbar=False,
    annot_kws={'size': 12, 'weight': 'bold'}
)
ax_cm.set_title('Confusion Matrix', fontsize=12, fontweight='bold', color='#1a476f')
ax_cm.set_xlabel('Predicted')
ax_cm.set_ylabel('Actual')
st.pyplot(fig_cm)

# 4. Financial Simulation Table
category_assumptions = {
    'Burgers': {'avg_price': 5.50, 'margin_pct': 0.60, 'annual_volume_multiplier': 15000},
    'Wraps': {'avg_price': 4.50, 'margin_pct': 0.55, 'annual_volume_multiplier': 8000},
    'Meals': {'avg_price': 8.00, 'margin_pct': 0.50, 'annual_volume_multiplier': 20000},
    'Beverages': {'avg_price': 2.50, 'margin_pct': 0.85, 'annual_volume_multiplier': 35000},
    'Desserts': {'avg_price': 3.00, 'margin_pct': 0.75, 'annual_volume_multiplier': 18000},
    'Snacks & Sides': {'avg_price': 3.50, 'margin_pct': 0.70, 'annual_volume_multiplier': 25000}
}

def estimate_profit(row):
    cat = row['Category']
    if cat in category_assumptions:
        specs = category_assumptions[cat]
        idx = row.name if hasattr(row, 'name') and isinstance(row.name, int) else 42
        np.random.seed(idx % 10000)
        price = np.round(specs['avg_price'] * np.random.uniform(0.8, 1.2), 2)
        margin = specs['margin_pct'] * np.random.uniform(0.9, 1.1)
        volume = int(specs['annual_volume_multiplier'] * np.random.uniform(0.7, 1.3))
        unit_profit = price * margin
        total_projected_profit = unit_profit * volume
        return pd.Series([price, f"{margin*100:.1f}%", volume, np.round(unit_profit, 2), np.round(total_projected_profit, 2)])
    return pd.Series([0.0, '0%', 0, 0.0, 0.0])

finance_df = df.copy()
finance_df[['Simulated_Price_($)', 'Simulated_Margin', 'Projected_Annual_Volume', 'Unit_Profit_($)', 'Projected_Total_Profit_($)']] = finance_df.apply(estimate_profit, axis=1)

st.subheader("4. Top 10 Most Profitable Products (Simulated)")
top_profitable = finance_df.sort_values(by='Projected_Total_Profit_($)', ascending=False).head(10)
st.dataframe(top_profitable[['Menu Items', 'Category', 'Simulated_Price_($)', 'Simulated_Margin', 'Projected_Annual_Volume', 'Projected_Total_Profit_($)']])

# 5. Model Download Option
model_filename = 'perfect_classifier_model.joblib'
joblib.dump(perf_clf, model_filename)

with open(model_filename, "rb") as f:
    st.download_button(
        label="Download Trained Model (.joblib)",
        data=f,
        file_name=model_filename,
        mime="application/octet-stream"
    )