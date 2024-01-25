import requests
import pandas as pd

url = 'http://localhost:9696/predict_serovar'

strain_profile_data = pd.read_csv('request_profile.csv', index_col=0)
strain_profile = strain_profile_data.iloc[0].to_dict()

response=requests.post(url,json=strain_profile).json()
print(response)

prediction = response['Serovar prediction']
print(f"Your Salmonella strain was classified as {prediction} based on its nutrient utilization profile")