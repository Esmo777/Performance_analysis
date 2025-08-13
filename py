import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# =======================================================
# 1. Configuration and Data Loading
# =======================================================

# Configure a professional and cheerful graphic style
plt.style.use('seaborn-v0_8-whitegrid')
vibrant_palette = sns.color_palette("husl", 8) # A harmonious and lively palette

# Create the directory for the graphs if it doesn't exist
output_dir = "graphs_for_github"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the dataset
df = pd.read_csv("HRDataset_v14.csv")

# Convert date columns to datetime objects
date_cols = ["DOB", "DateofHire", "DateofTermination", "LastPerformanceReview_Date"]
for col in date_cols:
    df[col] = pd.to_datetime(df[col], errors="coerce")

# Clean data for visualizations
# Fill missing values in PerformanceScore for the chart
df['PerformanceScore'] = df['PerformanceScore'].fillna('Not Classified')

# =======================================================
# 2. Descriptive Statistics and Outlier Detection
# =======================================================
print("====================================================")
print("Missing values per column:")
print(df.isnull().sum())
print("====================================================")

print("\nDescriptive statistics:")
print(df[["PerfScoreID", "Salary", "EngagementSurvey", "EmpSatisfaction",
           "SpecialProjectsCount", "DaysLateLast30", "Absences"]].describe())
print("====================================================")

# IQR method outlier detection function
def detect_outliers_iqr(data, col):
    Q1 = data[col].quantile(0.25)
    Q3 = data[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
    return outliers

outliers_salary = detect_outliers_iqr(df, "Salary")
outliers_absences = detect_outliers_iqr(df, "Absences")

print(f"\nNumber of Salary outliers: {len(outliers_salary)}")
print(f"Number of Absences outliers: {len(outliers_absences)}")
print("====================================================")

# =======================================================
# 3. Visualization of Distributions
# =======================================================
# Use subplots to display distributions in a grouped manner
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Performance Score Distributions', fontsize=18, fontweight='bold')

# PerfScoreID Distribution
sns.histplot(df["PerfScoreID"], kde=True, bins=10, color=vibrant_palette[0], ax=axes[0])
axes[0].set_title("PerfScoreID Distribution", fontsize=14)
axes[0].set_xlabel("PerfScoreID", fontsize=12)
axes[0].set_ylabel("Frequency", fontsize=12)

# PerformanceScore Distribution (more descriptive)
sns.countplot(data=df, x="PerformanceScore", palette=vibrant_palette, ax=axes[1], order=df['PerformanceScore'].value_counts().index)
axes[1].set_title("Distribution of Performance Scores", fontsize=14)
axes[1].set_xlabel("PerformanceScore", fontsize=12)
axes[1].set_ylabel("Number of Employees", fontsize=12)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(os.path.join(output_dir, "distribution_performance.png"))
plt.show()

# =======================================================
# 4. Comparisons by Key Variables
# =======================================================
# Performance by Department
plt.figure(figsize=(14, 7))
sns.boxplot(data=df, x="Department", y="PerfScoreID", palette=vibrant_palette)
plt.xticks(rotation=45)
plt.title("Performance by Department", fontsize=16, fontweight='bold')
plt.xlabel("Department", fontsize=12)
plt.ylabel("PerfScoreID", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "performance_by_department.png"))
plt.show()

# Performance by Sex
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x="Sex", y="PerfScoreID", palette=[vibrant_palette[1], vibrant_palette[5]])
plt.title("Performance by Sex", fontsize=16, fontweight='bold')
plt.xlabel("Sex", fontsize=12)
plt.ylabel("PerfScoreID", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "performance_by_sex.png"))
plt.show()

# =======================================================
# 5. Correlations
# =======================================================
# Select numerical columns for correlation
numerical_cols = ["PerfScoreID", "Salary", "EngagementSurvey", "EmpSatisfaction",
                  "SpecialProjectsCount", "DaysLateLast30", "Absences"]
corr_matrix = df[numerical_cols].corr()

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=.5)
plt.title("Correlation Heatmap", fontsize=16, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
plt.show()

# Scatter plot PerfScoreID vs EngagementSurvey
plt.figure(figsize=(10, 7))
sns.scatterplot(data=df, x="EngagementSurvey", y="PerfScoreID",
                hue="PerformanceScore", palette="viridis", s=100, style='PerformanceScore')
plt.title("PerfScoreID vs. EngagementSurvey", fontsize=16, fontweight='bold')
plt.xlabel("EngagementSurvey", fontsize=12)
plt.ylabel("PerfScoreID", fontsize=12)
plt.legend(title='Performance Score', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "scatterplot_engagement_performance.png"))
plt.show()

print("\n====================================================")
print("Insights Summary:")
print("====================================================")
print("- The majority of employees have a 'Fully Meets' performance score.")
print("- Some employees have a significantly higher salary than the average (outliers).")
print("- Extreme absences and late days concern a small number of employees.")
print("- Engagement and the number of special projects are positively correlated with performance.")
print("\nRecommendations:")
print("- Enhance employee engagement through internal programs.")
print("- Investigate cases of high absenteeism to understand the causes.")
print("- Encourage participation in special projects to boost performance.")
