# build_dataset.py

import pandas as pd
import numpy as np

# Set random seed for consistent results
np.random.seed(42)

# Settings
cities = ['Casablanca', 'Rabat', 'Marrakech', 'Agadir', 'Fes']
num_samples = 150

# Base price per city
base_price_dict = {
    'Casablanca': 700000,
    'Rabat': 650000,
    'Marrakech': 600000,
    'Agadir': 580000,
    'Fes': 500000
}

# Generate data
data = []
for _ in range(num_samples):
    city = np.random.choice(cities)
    size = np.random.randint(60, 300)
    bedrooms = np.random.randint(1, 6)
    bathrooms = np.random.randint(1, 4)
    garage = np.random.randint(0, 2)
    garden = np.random.randint(0, 2)
    balcony = np.random.randint(0, 2)
    year_built = np.random.randint(1970, 2024)
    age = 2025 - year_built
    noise = np.random.normal(0, 40000)

    base_price = base_price_dict[city]
    price = (
        base_price +
        size * 7000 +
        bedrooms * 40000 +
        bathrooms * 30000 +
        garage * 60000 +
        garden * 50000 +
        balcony * 20000 -
        age * 1000 +
        noise
    )

    data.append([
        city, size, bedrooms, bathrooms, garage, garden, balcony,
        year_built, round(price)
    ])

# Build DataFrame
columns = ['City', 'Size_m2', 'Bedrooms', 'Bathrooms', 'Garage', 'Garden', 'HasBalcony', 'YearBuilt', 'Price_MAD']
df = pd.DataFrame(data, columns=columns)

# Save to CSV
df.to_csv('realistic_house_data.csv', index=False)
print("✅ Dataset saved as realistic_house_data.csv")
