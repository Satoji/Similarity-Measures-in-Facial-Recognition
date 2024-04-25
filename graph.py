import matplotlib.pyplot as plt

# Thresholds for each distance measure
allThresholds = {
    'euclidean': [49.507, 76.677, 103.847, 131.017, 158.187],
    'cosine': [0.0989, 0.284, 0.468, 0.653, 0.838],
    'manhattan': [1206.346, 1978.246, 2750.149, 3522.049, 4293.950],
    'chebyshev': [7.563, 12.244, 16.926, 21.607, 26.288]
}

# FAR and FRR values for each distance measure at each threshold
FAR_values = {
    'euclidean': [0.0, 0.0, 0.0, 0.37, 1.0],
    'cosine': [0.0, 0.0, 0.0, 0.213, 1.0],
    'manhattan': [0.0, 0.0, 0.0, 0.31, 1.0],
    'chebyshev': [0.0, 0.0133, 0.5033, 0.9367, 0.99]
}

FRR_values = {
    'euclidean': [1.0, 0.6, 0.1, 0.0333, 0.0],
    'cosine': [1.0, 0.5, 0.0667, 0.0333, 0.0],
    'manhattan': [1.0, 0.6, 0.0667, 0.0333, 0.0],
    'chebyshev': [1.0, 0.4667, 0.0667, 0.0333, 0.0]
}


# Function to calculate EER
def calculate_EER(FAR, FRR):
    EER = []
    for i in range(len(FAR)):
        eer = (FAR[i] + FRR[i]) / 2
        EER.append(eer)
    return EER

# Calculate EER for each distance measure
EER_values = {}
for measure in FAR_values:
    FAR = FAR_values[measure]
    FRR = FRR_values[measure]
    EER_values[measure] = calculate_EER(FAR, FRR)

# Plotting EER for each distance measure
for measure in EER_values:
    plt.figure(figsize=(8, 6))
    plt.plot(allThresholds[measure], EER_values[measure])
    plt.xlabel('Threshold')
    plt.ylabel('Equal Error Rate (EER)')
    plt.title(f'Equal Error Rate (EER) for {measure.capitalize()}')
    plt.grid(True)
    plt.show()

# Plotting FAR and FRR for each distance measure
for measure in FAR_values:
    plt.figure(figsize=(8, 6))
    plt.plot(allThresholds[measure], FAR_values[measure], label='FAR')
    plt.plot(allThresholds[measure], FRR_values[measure], label='FRR')
    plt.xlabel('Threshold')
    plt.ylabel('Rate')
    plt.title(f'FAR and FRR for {measure.capitalize()}')
    plt.legend()
    plt.grid(True)
    plt.show()