"""
Warpage Data Analysis Script
Analyzes 50 warpage measurements from JSON file
"""

import json
import statistics

# Load the data
with open('data/uploads/leesihun/20251013_stats.json', 'r') as f:
    data = json.load(f)

files = data['files']
print("="*80)
print("WARPAGE ANALYSIS REPORT")
print("="*80)
print(f"\nTotal files analyzed: {len(files)}")
print(f"Source PDF: {data['source_pdf']}\n")

# Question 1: Highest maximum warpage value
max_values = [(f['file_id'], f['filename'], f['max']) for f in files]
highest_max = max(max_values, key=lambda x: x[2])
print("1. HIGHEST MAXIMUM WARPAGE VALUE:")
print(f"   File ID: {highest_max[0]}")
print(f"   Filename: {highest_max[1]}")
print(f"   Maximum value: {highest_max[2]}")

# Question 2: Lowest minimum warpage value
min_values = [(f['file_id'], f['filename'], f['min']) for f in files]
lowest_min = min(min_values, key=lambda x: x[2])
print(f"\n2. LOWEST MINIMUM WARPAGE VALUE:")
print(f"   File ID: {lowest_min[0]}")
print(f"   Filename: {lowest_min[1]}")
print(f"   Minimum value: {lowest_min[2]}")

# Question 3: Average mean warpage across all 50 measurements
mean_values = [f['mean'] for f in files]
average_mean = statistics.mean(mean_values)
print(f"\n3. AVERAGE MEAN WARPAGE ACROSS ALL 50 MEASUREMENTS:")
print(f"   Average mean: {average_mean:.4f}")

# Question 4: Overall standard deviation range
std_values = [f['std'] for f in files]
min_std = min(std_values)
max_std = max(std_values)
print(f"\n4. OVERALL STANDARD DEVIATION RANGE:")
print(f"   Minimum std: {min_std:.4f}")
print(f"   Maximum std: {max_std:.4f}")
print(f"   Range: {max_std - min_std:.4f}")

# Question 5: Measurement with most variability (highest range)
range_values = [(f['file_id'], f['filename'], f['range']) for f in files]
highest_range = max(range_values, key=lambda x: x[2])
print(f"\n5. MEASUREMENT WITH MOST VARIABILITY (HIGHEST RANGE):")
print(f"   File ID: {highest_range[0]}")
print(f"   Filename: {highest_range[1]}")
print(f"   Range: {highest_range[2]}")

# Question 6: Average kurtosis value
kurtosis_values = [f['kurtosis'] for f in files]
average_kurtosis = statistics.mean(kurtosis_values)
print(f"\n6. AVERAGE KURTOSIS VALUE ACROSS ALL MEASUREMENTS:")
print(f"   Average kurtosis: {average_kurtosis:.4f}")

# Question 7: Files with extreme kurtosis (>47)
extreme_kurtosis = [(f['file_id'], f['filename'], f['kurtosis'])
                    for f in files if f['kurtosis'] > 47]
extreme_kurtosis.sort(key=lambda x: x[2], reverse=True)

print(f"\n7. FILES WITH EXTREME KURTOSIS (>47) - POTENTIAL OUTLIERS:")
if extreme_kurtosis:
    print(f"   Found {len(extreme_kurtosis)} files with extreme kurtosis:\n")
    for file_id, filename, kurt in extreme_kurtosis:
        print(f"   {file_id}: {filename}")
        print(f"      Kurtosis: {kurt:.4f}\n")
else:
    print("   No files found with kurtosis > 47")

# Additional insights
print("="*80)
print("ADDITIONAL STATISTICAL INSIGHTS")
print("="*80)

print(f"\nMean warpage statistics:")
print(f"   Min mean: {min(mean_values):.4f}")
print(f"   Max mean: {max(mean_values):.4f}")
print(f"   Std dev of means: {statistics.stdev(mean_values):.4f}")

print(f"\nRange statistics:")
all_ranges = [f['range'] for f in files]
print(f"   Min range: {min(all_ranges):.4f}")
print(f"   Max range: {max(all_ranges):.4f}")
print(f"   Average range: {statistics.mean(all_ranges):.4f}")

print(f"\nKurtosis statistics:")
print(f"   Min kurtosis: {min(kurtosis_values):.4f}")
print(f"   Max kurtosis: {max(kurtosis_values):.4f}")
print(f"   Std dev of kurtosis: {statistics.stdev(kurtosis_values):.4f}")

print(f"\nSkewness statistics:")
skewness_values = [f['skewness'] for f in files]
print(f"   Min skewness: {min(skewness_values):.4f}")
print(f"   Max skewness: {max(skewness_values):.4f}")
print(f"   Average skewness: {statistics.mean(skewness_values):.4f}")

print("\n" + "="*80)
print("END OF REPORT")
print("="*80)
