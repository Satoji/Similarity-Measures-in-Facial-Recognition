import matplotlib.pyplot as plt

# Average threshold values calculated from Excel Sheet attached
allThresholds = {
    'euclidean': [30.10294418, 45.40843372, 60.71392326, 76.0194128, 91.32490234],
    'cosine': [0.05700353384, 0.1257356226, 0.1944677114, 0.2631998003, 0.3319318891],
    'manhattan': [273.1962585, 414.823378, 556.4504974, 698.0776169, 839.7047363],
    'chebyshev': [6.798832703, 11.15841703, 15.51800137, 19.8775857, 24.23717003]
}

FAR_values = {
    'euclidean': [0.0, 0.25, 0.6967, 0.9433, 1.0],
    'cosine': [0.0, 0.3433, 0.82, 0.9833, 1.0],
    'manhattan': [0.0, 0.2433, 0.6967, 0.9567, 1.0],
    'chebyshev': [0.0, 0.1866, 0.7833, 0.94, 1.0]
}

FRR_values = {
    'euclidean': [0.9667, 0.6667, 0.1667, 0.0, 0.0],
    'cosine': [1.0, 0.6, 0.2, 0.0, 0.0],
    'manhattan': [0.9667, 0.6667, 0.1667, 0.0, 0.0],
    'chebyshev': [1.0, 0.6667, 0.1, 0.0, 0.0]
}

def calculate_EER(FAR, FRR):
    EER = []
    for i in range(len(FAR)):
        eer = (FAR[i] + FRR[i]) / 2
        EER.append(eer)
    return EER

EER_values = {}
for measure in FAR_values:
    FAR = FAR_values[measure]
    FRR = FRR_values[measure]
    EER_values[measure] = calculate_EER(FAR, FRR)

for measure in EER_values:
    plt.figure(figsize=(8, 6))
    plt.plot(allThresholds[measure], EER_values[measure])
    plt.xlabel('Threshold')
    plt.ylabel('Equal Error Rate (EER)')
    plt.title(f'Equal Error Rate (EER) for {measure.capitalize()}')
    plt.grid(True)
    plt.show()

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