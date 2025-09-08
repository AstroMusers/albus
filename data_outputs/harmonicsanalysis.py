import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

df1 = pd.read_csv('data_outputs/injected_transits_output1.csv', header=0)
df2 = pd.read_csv('data_outputs/noninjected_transits_output1.csv', header=0)

dt = pd.read_csv('data_outputs/harmonics.csv', header=0)

count = 0

def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

HarmPower = np.array([])

for index, rows in dt.iterrows():
    HarmPower = np.append(HarmPower, rows['Power'])

IDs = []
RMSEs = []
correlations = []

for index, rows in df1.iterrows():
    PowerRatio = np.array([])
    PeriodRatio = np.array([])
    Powers = np.array((rows['Powers'])[1:-1].split(','), dtype=np.float32)
    Periods = np.array((rows['Periods'])[1:-1].split(','), dtype=np.float32)
    for i in range(len(Powers)):
        for j in range(len(Powers)):
            if i != j:
                PowerRatio = np.append(PowerRatio, (float(Powers[i])/float(Powers[j])))
                PeriodRatio = np.append(PeriodRatio, (float(Periods[i])/float(Periods[j])))
    if float(Periods[0]) < float(Periods[1]):
        print(f'flipping {rows["ID"]}')
        PowerRatio = 1/PowerRatio
    # plt.plot([0, 5], [0, 5], 'r--')

    # plt.plot(PeriodRatio, PowerRatio, 'o')
    # plt.xlabel('Period Ratio')
    # plt.ylabel('Power Ratio')
    # plt.title(f'Period Ratio vs Power Ratio of Periodogram Peaks for Sample Injection')
    # plt.show()
    correlations = np.append(correlations, (np.corrcoef(PowerRatio, PeriodRatio)[0, 1]))
    IDs = np.append(IDs, float(rows['ID']))
    RMSEs = np.append(RMSEs, RMSE(HarmPower, PowerRatio))

# plt.hist(correlations, bins=50, color='green', edgecolor='black', label='Injections', cumulative=False, alpha=0.5)
# plt.xlabel('Correlation Coefficient')
# plt.ylabel('Frequency')
# plt.title('Correlation Coefficient Distribution')

# plt.hist(RMSEs, bins=50, color='blue', edgecolor='black', label='Injections', alpha=0.5, cumulative=False)
# plt.xlabel('RMSE')
# plt.ylabel('Frequency')
# plt.title('RMSE Distribution')

threshold = -0.27
# count1 = np.where(RMSEs < threshold)[0].size
count1 = np.where(correlations > threshold)[0].size
# count1 = np.count_nonzero(correlations > threshold)


RMSEs = []
correlations = []

for index, rows in df2.iterrows():
    PowerRatio = []
    PeriodRatio = []
    Powers = np.array((rows['Powers'])[1:-1].split(','), dtype=np.float32)
    Periods = np.array((rows['Periods'])[1:-1].split(','), dtype=np.float32)
    for i in range(len(Powers)):
        for j in range(len(Powers)):
            if i != j:
                # print(f'Ratio P_{i}/P_{j}: ', float(Powers[i])/float(Powers[j]))
                PowerRatio = np.append(PowerRatio, (float(Powers[i])/float(Powers[j])))
                PeriodRatio = np.append(PeriodRatio, (float(Periods[i])/float(Periods[j])))
    if float(Periods[0]) < float(Periods[1]):
        print(f'flipping {rows["ID"]}')
        PowerRatio = 1/PowerRatio
    correlations = np.append(correlations, (np.corrcoef(PowerRatio, PeriodRatio)[0, 1]))
    IDs = np.append(IDs, float(rows['ID']))
    # plt.plot(PeriodRatio, PowerRatio, 'o')
    # plt.show()
    RMSEs = np.append(RMSEs, RMSE(HarmPower, PowerRatio))

# plt.hist(correlations, bins=50, color='blue', edgecolor='black', label='No Injections', alpha=0.5, cumulative=False)
# plt.xlabel('Correlation Coefficient')
# plt.ylabel('Frequency')
# plt.title('Correlation Coefficient Distribution')
# plt.legend()
# plt.axvline(threshold, color='black', linestyle='dashed', linewidth=1, label='Threshold')
# plt.show()

# plt.hist(RMSEs, bins=12, color='red', edgecolor='black', alpha=0.5, label='No Injections', cumulative=False)
# plt.legend()
# plt.axvline(threshold, color='black', linestyle='dashed', linewidth=1, label='Threshold')
# plt.show()

# count2 = np.where(RMSEs < threshold)[0].size
# count2 = np.count_nonzero(correlations > threshold)
count2 = np.where(correlations > threshold)[0].size

print('Injections:', count1)
print('No Injections:', count2)

# RMSEs = RMSEs.reshape(5, 5, 5)
# IDs = IDs.reshape(5, 5, 5)

# # print(RMSEs)
# # print(IDs)

# print(IDs[3])


# plt.imshow(correlations[3], interpolation='nearest')
# plt.xticks(ticks=[0, 1, 2, 3, 4], labels=[82, 84, 86, 88, 90])
# plt.yticks(ticks=[0, 1, 2, 3, 4], labels=[1, 3, 5, 7, 9])
# plt.xlabel('Inclination (deg)')
# plt.ylabel('Period (days)')
# plt.title('Correlation Coefficient Heatmap for Radius = 0.4 R$_J$')
# plt.colorbar()
# plt.show()

# radii = np.array([])
# periods = np.array([])
# inclin = np.array([])

# for i in range(5):
#     for j in range(5):
#         for k in range(5):
#             radii = np.append(radii, 1-(1/5)*i)
#             periods = np.append(periods, 1+(10/5)*j)
#             inclin = np.append(inclin, 90-(10/5)*k)

# correlations = correlations.reshape(5, 5, 5)
# x, y, z = np.indices(correlations.shape)

# x_flat = x.flatten()
# y_flat = y.flatten()
# z_flat = z.flatten()
# values = correlations.flatten()  # Corresponding values for coloring


# Cen3D = plt.figure()
# ax = Cen3D.add_subplot(111, projection='3d')

# sc = ax.scatter(x_flat, y_flat, z_flat, c=values, cmap='viridis', s=100)

# # Add color bar to show the scale
# colorbar = plt.colorbar(sc, ax=ax, pad=0.1)
# colorbar.set_label('Value')

# Set axis labels
# ax.set_xlabel('Inclination')
# ax.set_ylabel('Period')
# ax.set_zlabel('Radius')

# plt.xticks(ticks=[0, 1, 2, 3, 4], labels=[90, 88, 86, 84, 82])
# plt.yticks(ticks=[0, 1, 2, 3, 4], labels=[1, 3, 5, 7, 9])
# ax.set_zticks([0, 1, 2, 3, 4])
# ax.set_zticklabels([1, 0.8, 0.6, 0.4, 0.2])
# plt.title('Correlation Coefficient Heatmap')
# plt.show()

# print(IDs)


# num = 7
# Powers = np.array((df.loc[num,:]['Powers'])[1:-1].split(','), dtype=np.float32)
# # Periods = np.array(((df.loc[num,:]['Periods'])[1:-1].split(',')), dtype=np.float32)

# for i in range(len(Powers)):
#     for j in range(len(Powers)):
#         if i != j:
#             print(f'Ratio P_{i}/P_{j}: ', float(Powers[i])/float(Powers[j]))
#             PowerRatio = np.append(PowerRatio, float(Powers[i])/float(Powers[j]))
#             # PeriodRatio = np.append(PeriodRatio, float(Periods[i])/float(Periods[j]))

# print(PowerRatio)
# # print(PeriodRatio)
# print(HarmPower)

# print(RMSE(PowerRatio, HarmPower))

# plt.plot(PowerRatio, HarmPower, 'o')
# plt.xlabel('Power Ratio')
# plt.ylabel('Harmonic Power')
# plt.title('Power Ratio vs Harmonic Power')
# plt.plot(PowerRatio, HarmPower, 'o')
# plt.plot([0, 5], [0, 5], 'r--')
# plt.show()
