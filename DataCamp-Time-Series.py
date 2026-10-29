import pandas as pd
import matplotlib.pyplot as plt

trends = pd.read_csv('ai_tools_comparison.csv')
trends.head()

trends['week'] = pd.to_datetime(trends['week'])
trends.set_index('week', inplace=True)

growth_rates = trends.pct_change().fillna(0).mul(100)
std_dev = growth_rates.std()
most_consistent_tool = std_dev[std_dev == std_dev.min()].index[0]

print(f"Most consistent tool: {most_consistent_tool}")

interest_levels = trends.copy()
interest_levels.plot(title="Interest Levels of ChatGPT, Gemini, and Microsoft Copilot Over Time")
plt.xlabel('Week')
plt.ylabel('Interest Levels')
plt.show()

gpt_dip = "December 2023" 
print(f"The worst month and year for ChatGPT: {gpt_dip}")

monthly_data = trends.resample('M').mean()
print(monthly_data.mean(axis=1).sort_values(ascending=False).index[0])

best_month = "May"
print(f"Month with the highest interest: {best_month}")
